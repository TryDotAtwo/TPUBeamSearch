// Read-only oracle adapter: compile against the selected original GPU project's
// host/device helpers. This runs on the CPU and does not claim CUDA execution.
#include "hash.hpp"
#include "state.hpp"
#include "stream4.hpp"
#include "stream3.hpp"
#include <string>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "k1keys") {
        unsigned count, buckets;
        if (!(std::cin >> count >> buckets) || buckets == 0
            || (buckets & (buckets - 1)) != 0 || buckets > (1U << 30)) return 2;
        for (unsigned i = 0; i < count; ++i) {
            std::uint64_t w[4];
            for (auto& x : w) std::cin >> x;
            if (!std::cin) return 3;
            const beam::Hash128 hash{w[0] | (w[1] << 32), w[2] | (w[3] << 32)};
            std::cout << beam::hash128_fingerprint32(hash) << ' '
                      << (static_cast<std::uint32_t>(beam::hash128_bucket_key_0(hash)) & (buckets - 1)) << ' '
                      << (static_cast<std::uint32_t>(beam::hash128_bucket_key_1(hash)) & (buckets - 1)) << '\n';
        }
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "route") {
        unsigned count, world, shards;
        if (!(std::cin >> count >> world >> shards)
            || world == 0 || world > 256 || shards == 0) return 2;
        for (unsigned i = 0; i < count; ++i) {
            std::uint64_t w[4];
            for (auto& x : w) std::cin >> x;
            if (!std::cin) return 3;
            const beam::Hash128 hash{w[0] | (w[1] << 32), w[2] | (w[3] << 32)};
            std::cout << unsigned(beam::owner_from_hash128(hash, world)) << ' '
                      << beam::shard_from_hash128(hash, shards) << '\n';
        }
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "stream3") {
        unsigned count, threshold, rank, world;
        if (!(std::cin >> count >> threshold >> rank >> world)
            || world == 0 || world > 256 || rank >= world) return 2;
        std::vector<beam::Stream3CandidateInput> input(count);
        for (auto& m : input) {
            std::uint64_t w[9];
            for (auto& x : w) std::cin >> x;
            m = {{w[0] | (w[1] << 32), w[2] | (w[3] << 32)},
                 std::uint32_t(w[6]), std::uint32_t(w[8]),
                 w[4] | (w[5] << 32), std::uint8_t(w[7])};
        }
        if (!std::cin) return 3;
        const auto result = beam::stream3_threshold_dedup_split(input, threshold,
            std::uint16_t(rank), world);
        std::cout << "STREAM3\n" << result.local_pending.size() << ' '
                  << result.remote_send.size() << '\n';
        for (auto x : result.send_count) std::cout << x << ' ';
        std::cout << '\n';
        for (auto x : result.send_offset) std::cout << x << ' ';
        std::cout << '\n';
        for (const auto* group : {&result.local_pending, &result.remote_send}) {
            for (const auto& m : *group) {
                std::cout << (m.hash.lo & 0xffffffffULL) << ' ' << (m.hash.lo >> 32) << ' '
                          << (m.hash.hi & 0xffffffffULL) << ' ' << (m.hash.hi >> 32) << ' '
                          << (m.parent_idx & 0xffffffffULL) << ' ' << (m.parent_idx >> 32) << ' '
                          << m.score_key << ' ' << m.route_packed << '\n';
            }
        }
        return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "dedup") {
        unsigned count, threshold;
        if (!(std::cin >> count >> threshold)) return 2;
        std::vector<beam::CandidateMeta> input(count);
        for (auto& m : input) {
            std::uint64_t w[8];
            for (auto& x : w) std::cin >> x;
            m = {{w[0] | (w[1] << 32), w[2] | (w[3] << 32)},
                 w[4] | (w[5] << 32), std::uint32_t(w[6]), std::uint32_t(w[7])};
        }
        if (!std::cin) return 3;
        for (const auto& m : beam::stream4_threshold_sort_dedup(input, threshold)) {
            std::cout << (m.hash.lo & 0xffffffffULL) << ' ' << (m.hash.lo >> 32) << ' '
                      << (m.hash.hi & 0xffffffffULL) << ' ' << (m.hash.hi >> 32) << ' '
                      << (m.parent_idx & 0xffffffffULL) << ' ' << (m.parent_idx >> 32) << ' '
                      << m.score_key << ' ' << m.route_packed << '\n';
        }
        return 0;
    }
    unsigned batches, moves, classes;
    if (!(std::cin >> batches >> moves >> classes) || classes > beam::STATE_VALUE_PAD) return 2;
    std::vector<beam::State128> parents(batches);
    std::vector<beam::Generator> generators(moves);
    beam::State128 central{};
    unsigned value;
    for (auto& parent : parents) for (auto& x : parent.v) { std::cin >> value; x = value; }
    for (auto& generator : generators) for (auto& x : generator) { std::cin >> value; x = value; }
    for (auto& x : central.v) { std::cin >> value; x = value; }
    beam::ZobristTable table{};
    for (unsigned p = 0; p < beam::STATE_STORAGE_LEN; ++p) {
        for (unsigned c = 0; c < classes; ++c) {
            std::uint64_t w0, w1, w2, w3;
            std::cin >> w0 >> w1 >> w2 >> w3;
            table[p][c] = {w0 | (w1 << 32), w2 | (w3 << 32)};
        }
    }
    if (!std::cin) return 3;
    for (auto& parent : parents) for (auto& generator : generators) {
        const auto child = beam::apply_move(parent, generator);
        const auto hash = beam::hash_state(child, table);
        std::cout << (hash.lo & 0xffffffffULL) << ' ' << (hash.lo >> 32) << ' '
                  << (hash.hi & 0xffffffffULL) << ' ' << (hash.hi >> 32) << ' '
                  << beam::is_goal_state(child, central) << ' '
                  << unsigned(beam::owner_from_hash128(hash, 8)) << ' '
                  << beam::shard_from_hash128(hash, 7) << '\n';
    }
}
