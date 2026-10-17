#pragma once

#include "../status.hpp"

#include <cstddef>
#include <memory>

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#include <vector>
#endif

namespace apex::gemm {

using ThreadPoolTask = Status (*)(
    void* context,
    std::size_t worker_index,
    std::size_t worker_count) noexcept;

#if defined(__unix__) || defined(__APPLE__)

class ThreadPool {
public:
    [[nodiscard]] static ThreadPool& instance() noexcept {
        static ThreadPool pool;
        return pool;
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // The count includes the calling thread, which always runs worker zero.
    [[nodiscard]] Status initialize(std::size_t requested_worker_count) noexcept {
        if (requested_worker_count == 0) {
            return Status::InvalidArgument;
        }
        if (initialization_status_ != Status::Success) {
            return initialization_status_;
        }

        if (pthread_mutex_lock(&mutex_) != 0) {
            return Status::InternalError;
        }
        while (job_running_ && !stopping_) {
            if (pthread_cond_wait(&completion_condition_, &mutex_) != 0) {
                pthread_mutex_unlock(&mutex_);
                return Status::InternalError;
            }
        }
        if (stopping_) {
            pthread_mutex_unlock(&mutex_);
            return Status::Unsupported;
        }

        requested_worker_count_ =
            requested_worker_count > requested_worker_count_
            ? requested_worker_count
            : requested_worker_count_;
        const std::size_t target_background_workers = requested_worker_count_ - 1;
        try {
            workers_.reserve(target_background_workers);
        } catch (...) {
            pthread_mutex_unlock(&mutex_);
            return Status::AllocationFailed;
        }

        while (workers_.size() < target_background_workers) {
            std::unique_ptr<WorkerRecord> worker;
            try {
                worker = std::make_unique<WorkerRecord>();
            } catch (...) {
                pthread_mutex_unlock(&mutex_);
                return workers_.empty() ? Status::AllocationFailed : Status::Success;
            }
            worker->pool = this;
            worker->index = workers_.size() + 1;
            workers_.push_back(std::move(worker));
            WorkerRecord& worker_record = *workers_.back();
            const int create_status = pthread_create(
                &worker_record.thread, nullptr,
                &ThreadPool::worker_entry, &worker_record);
            if (create_status != 0) {
                workers_.pop_back();
                break;
            }
            worker_record.started = true;
        }

        pthread_mutex_unlock(&mutex_);
        return Status::Success;
    }

    [[nodiscard]] Status dispatch(
        ThreadPoolTask task,
        void* context) noexcept {
        if (task == nullptr) {
            return Status::InvalidArgument;
        }
        if (initialization_status_ != Status::Success) {
            return initialization_status_;
        }
        if (pthread_mutex_lock(&mutex_) != 0) {
            return Status::InternalError;
        }
        if (stopping_) {
            pthread_mutex_unlock(&mutex_);
            return Status::Unsupported;
        }

        // A concurrent caller owns its own job and runs synchronously.
        if (job_running_) {
            pthread_mutex_unlock(&mutex_);
            return task(context, 0, 1);
        }
        if (workers_.empty()) {
            pthread_mutex_unlock(&mutex_);
            return task(context, 0, 1);
        }

        job_ = task;
        job_context_ = context;
        job_status_ = Status::Success;
        completed_workers_ = 0;
        job_worker_count_ = workers_.size() + 1;
        job_running_ = true;
        ++generation_;
        if (pthread_cond_broadcast(&work_condition_) != 0) {
            job_running_ = false;
            job_ = nullptr;
            job_context_ = nullptr;
            pthread_mutex_unlock(&mutex_);
            return Status::InternalError;
        }
        pthread_mutex_unlock(&mutex_);

        const Status caller_status = task(context, 0, job_worker_count_);

        if (pthread_mutex_lock(&mutex_) != 0) {
            return Status::InternalError;
        }
        record_status(caller_status);
        while (completed_workers_ < workers_.size()) {
            if (pthread_cond_wait(&completion_condition_, &mutex_) != 0) {
                job_status_ = Status::InternalError;
                break;
            }
        }
        const Status result = job_status_;
        job_running_ = false;
        job_ = nullptr;
        job_context_ = nullptr;
        pthread_cond_broadcast(&completion_condition_);
        pthread_mutex_unlock(&mutex_);
        return result;
    }

    [[nodiscard]] std::size_t worker_count() noexcept {
        if (initialization_status_ != Status::Success ||
            pthread_mutex_lock(&mutex_) != 0) {
            return 1;
        }
        const std::size_t count = workers_.size() + 1;
        pthread_mutex_unlock(&mutex_);
        return count;
    }

    [[nodiscard]] Status shutdown() noexcept {
        if (initialization_status_ != Status::Success || stopped_) {
            return initialization_status_;
        }
        if (pthread_mutex_lock(&mutex_) != 0) {
            return Status::InternalError;
        }
        while (job_running_) {
            if (pthread_cond_wait(&completion_condition_, &mutex_) != 0) {
                pthread_mutex_unlock(&mutex_);
                return Status::InternalError;
            }
        }
        stopping_ = true;
        pthread_cond_broadcast(&work_condition_);
        pthread_mutex_unlock(&mutex_);

        Status result = Status::Success;
        for (const auto& worker : workers_) {
            if (worker->started && pthread_join(worker->thread, nullptr) != 0) {
                result = Status::InternalError;
            }
            worker->started = false;
        }
        workers_.clear();
        stopped_ = true;
        return result;
    }

    ~ThreadPool() noexcept {
        (void)shutdown();
        destroy_synchronization();
    }

private:
    struct WorkerRecord {
        pthread_t thread{};
        ThreadPool* pool = nullptr;
        std::size_t index = 0;
        bool started = false;
        std::size_t observed_generation = 0;
    };

    ThreadPool() noexcept {
        if (pthread_mutex_init(&mutex_, nullptr) != 0) {
            initialization_status_ = Status::InternalError;
            return;
        }
        mutex_initialized_ = true;
        if (pthread_cond_init(&work_condition_, nullptr) != 0) {
            initialization_status_ = Status::InternalError;
            return;
        }
        work_condition_initialized_ = true;
        if (pthread_cond_init(&completion_condition_, nullptr) != 0) {
            initialization_status_ = Status::InternalError;
            return;
        }
        completion_condition_initialized_ = true;
    }

    static void* worker_entry(void* argument) noexcept {
        auto* worker = static_cast<WorkerRecord*>(argument);
        if (worker == nullptr || worker->pool == nullptr) {
            return nullptr;
        }
        worker->pool->worker_loop(*worker);
        return nullptr;
    }

    void worker_loop(WorkerRecord& worker) noexcept {
        if (pthread_mutex_lock(&mutex_) != 0) {
            return;
        }
        while (!stopping_) {
            while (!stopping_ && worker.observed_generation == generation_) {
                if (pthread_cond_wait(&work_condition_, &mutex_) != 0) {
                    stopping_ = true;
                    pthread_cond_broadcast(&completion_condition_);
                    pthread_mutex_unlock(&mutex_);
                    return;
                }
            }
            if (stopping_) {
                break;
            }

            worker.observed_generation = generation_;
            const ThreadPoolTask task = job_;
            void* const context = job_context_;
            const std::size_t count = job_worker_count_;
            pthread_mutex_unlock(&mutex_);
            const Status task_status = task == nullptr
                ? Status::InternalError
                : task(context, worker.index, count);
            if (pthread_mutex_lock(&mutex_) != 0) {
                return;
            }
            record_status(task_status);
            ++completed_workers_;
            if (completed_workers_ == workers_.size()) {
                pthread_cond_broadcast(&completion_condition_);
            }
        }
        pthread_mutex_unlock(&mutex_);
    }

    void record_status(Status status) noexcept {
        if (job_status_ == Status::Success && status != Status::Success) {
            job_status_ = status;
        }
    }

    void destroy_synchronization() noexcept {
        if (completion_condition_initialized_) {
            pthread_cond_destroy(&completion_condition_);
            completion_condition_initialized_ = false;
        }
        if (work_condition_initialized_) {
            pthread_cond_destroy(&work_condition_);
            work_condition_initialized_ = false;
        }
        if (mutex_initialized_) {
            pthread_mutex_destroy(&mutex_);
            mutex_initialized_ = false;
        }
    }

    pthread_mutex_t mutex_{};
    pthread_cond_t work_condition_{};
    pthread_cond_t completion_condition_{};
    std::vector<std::unique_ptr<WorkerRecord>> workers_;
    ThreadPoolTask job_ = nullptr;
    void* job_context_ = nullptr;
    std::size_t job_worker_count_ = 1;
    std::size_t generation_ = 0;
    std::size_t completed_workers_ = 0;
    std::size_t requested_worker_count_ = 1;
    Status job_status_ = Status::Success;
    Status initialization_status_ = Status::Success;
    bool job_running_ = false;
    bool stopping_ = false;
    bool stopped_ = false;
    bool mutex_initialized_ = false;
    bool work_condition_initialized_ = false;
    bool completion_condition_initialized_ = false;
};

#else

// Platforms without pthread keep the API usable in caller-only mode.
class ThreadPool {
public:
    [[nodiscard]] static ThreadPool& instance() noexcept {
        static ThreadPool pool;
        return pool;
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] Status initialize(std::size_t requested_worker_count) noexcept {
        if (requested_worker_count == 0) {
            return Status::InvalidArgument;
        }
        if (stopped_) {
            return Status::Unsupported;
        }
        return Status::Success;
    }

    [[nodiscard]] Status dispatch(ThreadPoolTask task, void* context) noexcept {
        if (task == nullptr) {
            return Status::InvalidArgument;
        }
        if (stopped_) {
            return Status::Unsupported;
        }
        return task(context, 0, 1);
    }

    [[nodiscard]] std::size_t worker_count() const noexcept { return 1; }
    [[nodiscard]] Status shutdown() noexcept {
        stopped_ = true;
        return Status::Success;
    }

    ~ThreadPool() noexcept { (void)shutdown(); }

private:
    ThreadPool() noexcept = default;
    bool stopped_ = false;
};

#endif

}  // namespace apex::gemm
