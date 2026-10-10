#include "ggml-vulkan-tp5-dependency-ledger.h"
#include <cassert>
#include <cstdio>
int main(){
    vk_tp5_dependency_ledger d;d.note(false,true);d.note(true,false);d.note(true,false);
    d.note(false,true);d.note(false,true);d.note(true,false);d.finish();
    assert(d.dispatches==3&&d.epochs==2&&d.maximum_width==2);
    d.finish();assert(d.epochs==2); // empty boundaries never create time
    vk_tp5_dependency_ledger parallel;for(int i=0;i<100;++i)parallel.note(true,false);parallel.finish();
    assert(parallel.dispatches==100&&parallel.epochs==1&&parallel.maximum_width==100);
    vk_tp5_dependency_ledger serial;for(int i=0;i<100;++i){serial.note(true,false);serial.note(false,true);}serial.finish();
    assert(serial.dispatches==100&&serial.epochs==100&&serial.maximum_width==1);
    std::puts("explicit execution-dependency census: empty/parallel/serial epochs verified");
}
