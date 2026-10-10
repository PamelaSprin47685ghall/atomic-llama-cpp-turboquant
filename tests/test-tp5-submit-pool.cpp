#include "ggml-vulkan-tp5-submit-pool.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <set>

int main() {
    vk_tp5_submit_pool pool;
    vk_tp5_submit_pool::results out;
    std::array<std::thread::id,5> ids;
    std::atomic<int> entered{0};
    assert(pool.run([&](size_t i){ids[i]=std::this_thread::get_id();++entered;
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(entered.load()!=5&&std::chrono::steady_clock::now()<until)std::this_thread::yield();
        return entered.load()==5?(int)i:-99;},out));
    assert(std::set<std::thread::id>(ids.begin(),ids.end()).size()==5);
    for(size_t i=0;i<5;++i)assert(out[i]==(int)i);
    for(int run=0;run<100;++run) {
        assert(pool.run([&](size_t i){return i==(size_t)(run%5)?-4:0;},out));
        for(size_t i=0;i<5;++i)assert(out[i]==(i==(size_t)(run%5)?-4:0));
    }
    assert(pool.run([](size_t i)->int{if(i==2)throw 7;return 0;},out));assert(out[2]==-13);
    // Deadline expiry must still return all completed results and must not
    // leave a worker holding a reference to the caller's stack.
    std::atomic<int> retired{0};
    assert(!pool.run([&](size_t i){std::this_thread::sleep_for(std::chrono::milliseconds(10));++retired;return i==4?-4:0;},out,std::chrono::milliseconds(1)));
    assert(retired==5);for(size_t i=0;i<5;++i)assert(out[i]==(i==4?-4:0));
    assert(!pool.run([](size_t){return 0;},out));
    for(int code:out)assert(code==-13);
    std::puts("parallel queues: all5 execute concurrently;100 partial-result sets;exception;deadline join/lifetime checks passed");
}
