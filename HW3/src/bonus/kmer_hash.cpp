#include <cstdio>
#include <cstdlib>
#include <vector>
#include <list>
#include <set>
#include <numeric>
#include <cstddef>
#include <upcxx/upcxx.hpp>

#include "kmer_t.hpp"
#include "read_kmers.hpp"
#include "hash_map.hpp"

#include "butil.hpp"

HashMap* glob_h;

int glob_insert(const kmer_pair &kmer) {
    glob_h->queue(kmer);
    return 1;
}

int main(int argc, char **argv) {
  upcxx::init();


    /*
        1) test UPC to see how "shared/distributed" memory works
            - need to be able to BROADCAST a pointer from rank 0 to the world for things to be shared
        2) change hashtable class insert and find to partition the array backing the hashtable across n_proc arrays
        3) find a way to load balance the work (apparently the kmers are start kmers are approximately evenly distributed, so no global load balancing needed)
    */

  if (argc < 2) {
    BUtil::print("usage: srun -N nodes -n ranks ./kmer_hash kmer_file [verbose|test]\n");
    upcxx::finalize();
    exit(1);
  }

  std::string kmer_fname = std::string(argv[1]);
  std::string run_type = "";

  if (argc >= 3) {
    run_type = std::string(argv[2]);
  }

  int ks = kmer_size(kmer_fname);

  if (ks != KMER_LEN) {
    throw std::runtime_error("Error: " + kmer_fname + " contains " +
      std::to_string(ks) + "-mers, while this binary is compiled for " +
      std::to_string(KMER_LEN) + "-mers.  Modify packing.hpp and recompile.");
  }

  size_t n_kmers = line_count(kmer_fname);





  // Load factor of 0.5
  size_t hash_table_size = n_kmers * (1.0  * 3);
  hash_table_size = (hash_table_size + upcxx::rank_n() - 1) / upcxx::rank_n() * upcxx:: rank_n();

    upcxx::global_ptr<kmer_pair> local_data_copy = upcxx::new_array<kmer_pair>(hash_table_size);
    std::vector<upcxx::global_ptr<kmer_pair>> dist_data_ptrs;
    for (int i = 0; i < upcxx::rank_n(); i++) {
        dist_data_ptrs.push_back(upcxx::broadcast(local_data_copy, i).wait());
    }


  HashMap hashmap(hash_table_size, dist_data_ptrs);
    glob_h = & hashmap;
  if (run_type == "verbose") {
    BUtil::print("Initializing hash table of size %d for %d kmers.\n",
      hash_table_size, n_kmers);
  }

    // barrier for after init
  upcxx::barrier();

  std::vector <kmer_pair> kmers = read_kmers(kmer_fname, upcxx::rank_n(), upcxx::rank_me());

  if (run_type == "verbose") {
    BUtil::print("Finished reading kmers.\n");
  }

  auto start = std::chrono::high_resolution_clock::now();

  std::vector <kmer_pair> start_nodes;
  std::vector<upcxx::future<int>>  rpcs;
  for (auto &kmer : kmers) {
    uint64_t hash = kmer.hash();
    int node_num = (hash % hashmap.size()) / (hashmap.size() / upcxx::rank_n());
    rpcs.push_back(upcxx::rpc(node_num, glob_insert, kmer));
    //hashmap.insert(kmer);

    if (kmer.backwardExt() == 'F') {
      start_nodes.push_back(kmer);
    }
  }
    for (auto && r : rpcs) {
        r.wait();
    }
  auto end_insert = std::chrono::high_resolution_clock::now();
  upcxx::barrier();
    hashmap.reduce();
    upcxx::barrier();
  double insert_time = std::chrono::duration <double> (end_insert - start).count();
  if (run_type != "test") {
    BUtil::print("Finished inserting in %lf\n", insert_time);
  }
  upcxx::barrier();

  auto start_read = std::chrono::high_resolution_clock::now();

  std::list <std::list <kmer_pair>> contigs;
  for (const auto &start_kmer : start_nodes) {
    std::list <kmer_pair> contig;
    contig.push_back(start_kmer);
    while (contig.back().forwardExt() != 'F') {
      kmer_pair kmer;
      bool success = hashmap.find(contig.back().next_kmer(), kmer);
      if (!success) {
        throw std::runtime_error("Error: k-mer not found in hashmap.");
      }
      contig.push_back(kmer);
    }
    contigs.push_back(contig);
  }

  auto end_read = std::chrono::high_resolution_clock::now();
  upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration <double> read = end_read - start_read;
  std::chrono::duration <double> insert = end_insert - start;
  std::chrono::duration <double> total = end - start;

  int numKmers = std::accumulate(contigs.begin(), contigs.end(), 0,
    [] (int sum, const std::list <kmer_pair> &contig) {
      return sum + contig.size();
    });

  if (run_type != "test") {
    BUtil::print("Assembled in %lf total\n", total.count());
    BUtil::print("%d total ht accesses on %d tries\n", hashmap.tot_hits, hashmap.num_uses);
  }

  if (run_type == "verbose") {
    printf("Rank %d reconstructed %d contigs with %d nodes from %d start nodes."
      " (%lf read, %lf insert, %lf total)\n", upcxx::rank_me(), contigs.size(),
      numKmers, start_nodes.size(), read.count(), insert.count(), total.count());
  }

  if (run_type == "test") {
    std::ofstream fout("test_" + std::to_string(upcxx::rank_me()) + ".dat");
    for (const auto &contig : contigs) {
      fout << extract_contig(contig) << std::endl;
    }
    fout.close();
  }

  upcxx::finalize();
  return 0;
}