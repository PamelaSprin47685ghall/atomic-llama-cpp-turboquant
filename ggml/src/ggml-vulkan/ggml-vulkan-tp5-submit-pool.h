#pragma once
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

// One worker per DISTINCT VkQueue. The caller owns external synchronization.
// No worker mutates the communicator or timeline bookkeeping: the caller must
// retire EVERY successful submission, even when another rank fails.
class vk_tp5_submit_pool {
public:
    static constexpr size_t ranks=5;
    using results=std::array<int,ranks>;
    vk_tp5_submit_pool() {
        try {
            for(size_t i=0;i<ranks;++i) workers[i]=std::thread([this,i]{work(i);});
        } catch (...) {stop_and_join();throw;}
    }
    ~vk_tp5_submit_pool(){stop_and_join();}
    vk_tp5_submit_pool(const vk_tp5_submit_pool&)=delete;
    vk_tp5_submit_pool&operator=(const vk_tp5_submit_pool&)=delete;

    bool run(std::function<int(size_t)> call,results & out,
             std::chrono::milliseconds deadline=std::chrono::milliseconds(2000)) {
        out.fill(-13); // refusal means no native submission, never VK_SUCCESS
        std::unique_lock<std::mutex> lock(mutex);
        if(stop||completed!=ranks)return false;
        task=std::move(call);codes.fill(-13);completed=0;++epoch;ready.notify_all();
        const bool done=finished.wait_for(lock,deadline,[this]{return completed==ranks;});
        if(!done) {
            // Never return while accepted native submissions may still be
            // written into a caller-owned result array. Poison the pool, join
            // workers, then hand ALL real return codes back for native drain.
            stop=true;lock.unlock();ready.notify_all();join();lock.lock();
        }
        out=codes;task={};return done;
    }
private:
    std::array<std::thread,ranks> workers;
    std::mutex mutex;
    std::condition_variable ready,finished;
    std::function<int(size_t)> task;
    results codes{};
    size_t completed=ranks;
    uint64_t epoch=0;
    bool stop=false;
    void work(size_t rank) {
        uint64_t observed=0;
        std::unique_lock<std::mutex> lock(mutex);
        for(;;) {
            ready.wait(lock,[&]{return stop||epoch!=observed;});
            if(stop&&epoch==observed)return;
            observed=epoch;lock.unlock();
            int result=-13;
            try {result=task(rank);}catch(...){result=-13;}
            lock.lock();codes[rank]=result;++completed;
            if(completed==ranks)finished.notify_one();
            if(stop)return;
        }
    }
    void join(){for(auto & t:workers)if(t.joinable())t.join();}
    void stop_and_join(){
        {std::lock_guard<std::mutex> lock(mutex);stop=true;}
        ready.notify_all();join();
    }
};
