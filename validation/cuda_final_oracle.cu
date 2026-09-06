// Thin binary adapter to unchanged D:/100XH100 CUDA final functions.
#include <cuda_runtime.h>
#include "final_materialize.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Device {
    T* ptr=nullptr;
    explicit Device(std::size_t count) { check(cudaMalloc(&ptr,std::max<std::size_t>(count,1)*sizeof(T))); }
    ~Device() { if(ptr) cudaFree(ptr); }
    Device(const Device&)=delete;
};
static void read(std::ifstream& in,void* data,std::size_t size) {
    if(size && !in.read(static_cast<char*>(data),size)) throw std::runtime_error("truncated fixture");
}
int main(int argc,char** argv) {
    try {
        if(argc!=3 && argc!=4) throw std::runtime_error("usage: oracle input.bin output.bin [--local-slots]");
        const bool local_slots=argc==4 && std::strcmp(argv[3],"--local-slots")==0;
        if(argc==4 && !local_slots) throw std::runtime_error("unknown mode");
        std::ifstream in(argv[1],std::ios::binary);
        char magic[8]; read(in,magic,8);
        if(std::memcmp(magic,"TFIN0001",8)) throw std::runtime_error("bad magic");
        std::uint32_t h[6]; read(in,h,sizeof(h));
        const auto parents_count=h[0],count=h[1],target_count=h[5];
        if(!parents_count || parents_count>(1U<<20) || count>(1U<<20) ||
           h[2]!=beam::MOVE_COUNT || h[3]!=beam::STATE_LEN || h[4]!=beam::STATE_STORAGE_LEN)
            throw std::runtime_error("fixture ABI/size mismatch");
        std::vector<beam::State128> parents(parents_count);
        std::vector<beam::FinalRequest> requests(count);
        std::vector<std::uint8_t> generators(beam::MOVE_COUNT*beam::STATE_STORAGE_LEN);
        read(in,parents.data(),parents.size()*sizeof(parents[0]));
        read(in,requests.data(),requests.size()*sizeof(requests[0]));
        read(in,generators.data(),generators.size());
        if(in.peek()!=std::ifstream::traits_type::eof()) throw std::runtime_error("trailing fixture bytes");
        // Remote grouped requests do not obey target==input slot. The original
        // dispatcher calls its debug slot validator only on the local path.
        // Protect adapter memory access without rewriting the CUDA functions.
        for(const auto& request:requests) {
            if(request.parent_idx>=parents_count || request.target_local_idx>=target_count || request.move>=beam::MOVE_COUNT)
                throw std::runtime_error("request out of bounds");
        }
        for(std::size_t move=0;move<beam::MOVE_COUNT;++move) {
            std::vector<bool> seen(beam::STATE_LEN);
            for(std::size_t p=0;p<beam::STATE_LEN;++p) {
                auto value=generators[move*beam::STATE_STORAGE_LEN+p];
                if(value>=beam::STATE_LEN || seen[value]) throw std::runtime_error("invalid generator");
                seen[value]=true;
            }
        }
        check(cudaSetDevice(0));
        cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props,0));
        std::cout<<"device="<<props.name<<" sm="<<props.major<<props.minor<<"\n";
        Device<beam::State128> dp(parents_count);
        Device<beam::FinalRequest> dr(count);
        Device<std::uint8_t> dg(generators.size());
        Device<beam::FinalResponse> dout(count);
        Device<beam::FinalRequestValidationError> de(1);
        check(cudaMemcpy(dp.ptr,parents.data(),parents.size()*sizeof(parents[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(dg.ptr,generators.data(),generators.size(),cudaMemcpyHostToDevice));
        std::vector<beam::FinalResponse> result(count);
        if(count) {
            check(cudaMemcpy(dr.ptr,requests.data(),requests.size()*sizeof(requests[0]),cudaMemcpyHostToDevice));
            if(local_slots) beam::validate_final_requests_cuda(dr.ptr,count,parents_count,target_count,de.ptr,0);
            beam::final_materialize_responses_cuda(dp.ptr,dr.ptr,dg.ptr,dout.ptr,count,0);
            check(cudaDeviceSynchronize());
            check(cudaMemcpy(result.data(),dout.ptr,result.size()*sizeof(result[0]),cudaMemcpyDeviceToHost));
        }
        std::ofstream out(argv[2],std::ios::binary|std::ios::trunc);
        if(!out) throw std::runtime_error("cannot create output");
        if(count) out.write(reinterpret_cast<const char*>(result.data()),result.size()*sizeof(result[0]));
        if(!out) throw std::runtime_error("output write failed");
        std::cout<<"responses="<<count<<" bytes="<<result.size()*sizeof(beam::FinalResponse)<<"\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<"\n";
        return 2;
    }
}
