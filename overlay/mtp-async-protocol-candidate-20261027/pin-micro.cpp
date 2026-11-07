#include <sched.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
int main(){ cpu_set_t all,one; sched_getaffinity(0,sizeof(all),&all); CPU_ZERO(&one); CPU_SET(21,&one); std::vector<double> v;
 for(int i=0;i<2000;i++){ auto t=std::chrono::steady_clock::now(); sched_setaffinity(0,sizeof(one),&one); sched_setaffinity(0,sizeof(all),&all);
  v.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count()); }
 std::sort(v.begin(),v.end()); printf("{\"pin_unpin_roundtrip_us_p50\":%.2f,\"p99\":%.2f,\"max\":%.2f}\n",v[1000],v[1980],v.back()); }
