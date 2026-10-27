// Host-only micro: cost of the meet() rendezvous as implemented (condvar wait_for 1ms) vs busy-wait,
// and std::mutex vs FIFO spin ticket lock handoff, 2 threads (= 2 windows). No GPU.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
static inline void pause_cpu(){ __builtin_ia32_pause(); }
static uint64_t ns(){ return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct cv_pair { std::mutex m; std::condition_variable c; int arr=0; uint64_t gen=0; bool spin;
  void meet(){ std::unique_lock<std::mutex> l(m); auto g=gen; if(++arr==2){arr=0;++gen;c.notify_all();return;}
    while(gen==g){ if(spin){l.unlock();pause_cpu();l.lock();} else c.wait_for(l,std::chrono::milliseconds(1)); } } };
struct ticket { std::atomic<uint32_t> n{0}, s{0}; void lock(){auto me=n.fetch_add(1); while(s.load(std::memory_order_acquire)!=me) pause_cpu();} void unlock(){s.fetch_add(1,std::memory_order_release);} };
template<class F> void run(const char* name, F body){
  std::vector<double> lat[2]; std::atomic<uint64_t> stamp[2];
  std::thread t[2]; for(int i=0;i<2;i++) t[i]=std::thread([&,i]{ for(int k=0;k<2000;k++){ double d=body(i,k); if(k>50) lat[i].push_back(d);} });
  for(auto&x:t)x.join();
  auto v=lat[0]; v.insert(v.end(),lat[1].begin(),lat[1].end()); std::sort(v.begin(),v.end());
  printf("{\"case\":\"%s\",\"p50_us\":%.2f,\"p90_us\":%.2f,\"p99_us\":%.2f,\"max_us\":%.2f}\n",name,v[v.size()/2]/1e3,v[v.size()*9/10]/1e3,v[v.size()*99/100]/1e3,v.back()/1e3);
}
int main(){
  for (bool spin : {false,true}) {
    cv_pair p; p.spin=spin;
    // window 1 arrives 200us late every cycle: measure release latency after the last arrival
    std::atomic<uint64_t> last{0};
    run(spin?"meet_busywait":"meet_condvar_waitfor1ms",[&](int i,int k){ if(i==1){ auto t=ns(); while(ns()-t<200000) pause_cpu(); last.store(ns()); }
        p.meet(); uint64_t now=ns(); return i==0 ? double(now-last.load()) : 0.0; });
  }
  { std::mutex m; std::atomic<uint64_t> rel{0};
    run("std_mutex_handoff",[&](int i,int k){ m.lock(); auto t=ns(); double d = rel.load()? double(t-rel.load()):0; while(ns()-t<20000) pause_cpu(); rel.store(ns()); m.unlock(); return d; }); }
  { ticket m; std::atomic<uint64_t> rel{0};
    run("fifo_spin_handoff",[&](int i,int k){ m.lock(); auto t=ns(); double d = rel.load()? double(t-rel.load()):0; while(ns()-t<20000) pause_cpu(); rel.store(ns()); m.unlock(); return d; }); }
}
