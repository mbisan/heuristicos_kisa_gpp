#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;

typedef Eigen::MatrixXi Graph;
typedef Eigen::VectorXi Perm;

struct ls_output {
    Perm perm;
    int cost;
};

Perm loadPerm(string filename){
    ifstream file;
    file.open(filename);

    string line;
    if (file.is_open()){
        getline(file, line);
    } else {
        file.close();
        return Perm(0);
    }

    size_t len = stoi(line);

    cout << "Loading Partition of size " << len << endl;

    Perm perm = Perm::Constant(len, 0);

    getline(file, line);

    for (int i=0; i<len; i++) {
        perm(i) = stoi(line.substr(i, 1));
    }
    file.close();
    return perm;
}

Graph loadGraph(string filename){
    ifstream file;
    file.open(filename);

    string line;
    if (file.is_open()){
        getline(file, line);
    } else {
        file.close();
        return Graph(0, 0);
    }

    string delimiter = " ";

    size_t token = stoi(line.substr(0, line.find(delimiter)));
    line.erase(0, line.find(delimiter) + delimiter.length());

    Graph graph = Graph::Constant(token, token, 0);

    cout << "Loading Graph with " << token << " vertices and ";

    token = stoi(line.substr(0, line.find(delimiter)));

    cout << token << " edges" << endl;

    size_t counter = 0;
    while(getline(file, line)){
        // cout << line.length() <<endl;
        size_t pos = 0;
        while ((pos = line.find(delimiter)) != string::npos){
            size_t token = stoi(line.substr(0, line.find(delimiter)));
            line.erase(0, line.find(delimiter) + delimiter.length());

            // cout << "extract" << line.length() <<endl;
            // cout << "Edge " << counter << " " << token << endl;

            graph(counter, token - 1) = 1;
        }
        graph(counter, stoi(line) - 1) = 1;
        counter++;
    }
    file.close();
    return graph;
}

int saveResults(ls_output res, string filename) {
    ofstream pfile;
    pfile.open(filename + ".p");

    if (pfile.is_open()){
        cout << "Saving resulting partition to " + filename + ".p" << endl;
        int nvert = res.perm.size();
        pfile << nvert << endl;
        for (int i=0; i<nvert; i++){
            pfile << res.perm(i);
        }
    }
    pfile.close();
    return 0;
}

int cost(Graph graph, Perm perm){
    int res = 0;

    for (int i = 0; i < graph.rows(); i++){
        if(perm(i)){
            res += (Perm::Constant(perm.size(), 1) - perm).dot(graph.row(i));
        } else {
            res += perm.dot(graph.row(i));
        }
    }
    return res;
}

auto localSearch(Graph graph, Perm perm, int n) {
    int cost_p = cost(graph, perm);
    int recall = cost_p;
    int best_c = cost_p;

    Perm p = Perm::Constant(perm.size(), 0); p = perm;
    Perm new_p = Perm::Constant(perm.size(), 0); new_p = perm;
    Perm best_p = Perm::Constant(perm.size(), 0); best_p = perm;

    Perm ones = Perm::Constant(perm.size(), 1);
    Perm new_p_m(perm.size());

    for (int counter = 0; counter < n; counter++){
        Perm p_m = ones - p;

        for (int i=0; i<perm.size(); i++){
            if (!p(i)) continue;

            int sub_cause_i = graph.row(i).dot(p_m) + graph.col(i).dot(p_m);

            new_p(i) ^= 1;

            for (int j=0; j< perm.size(); j++){
                if (p(j)) continue;

                new_p(j) ^= 1;

                new_p_m = ones - new_p;

                cost_p -= graph.row(j).dot(p) + graph.col(j).dot(p) + sub_cause_i;

                cost_p += graph.row(j).dot(new_p_m) +
                          graph.col(j).dot(new_p_m) +
                          graph.row(i).dot(new_p) +
                          graph.col(i).dot(new_p);

                if (cost_p < best_c){
                    best_p = new_p;
                    best_c = cost_p;
                }

                cost_p = recall;
                new_p(j) ^= 1;
            }
            new_p(i) ^= 1;
        }
        p = best_p;
        new_p = best_p;
        recall = best_c;
        cost_p = recall;
    }

    return ls_output{best_p, best_c};
}

int main(int argc, char * argv[]){
    if (argc % 2 == 0 || argc == 1) {
        cout << "usage: (exec) -g GRAPH_FILE -p PARTITION_FILE -n ITERATIONS -o OUTPUT_FILE" << endl;
        return 0;
    }

    Graph test = (Eigen::ArrayXXd::Random(10, 10) > 0).cast<int>();
    Perm testp;
    int n = 1;
    string output = "test";

    for (int i=1; i<argc; i=i+2){
        if(strcmp(argv[i], "-g")==0){
            test = loadGraph(argv[i+1]);
        } else if (strcmp(argv[i], "-p")==0){
            testp = loadPerm(argv[i+1]);
        } else if (strcmp(argv[i], "-n")==0){
            n = stoi(argv[i+1]);
        } else if (strcmp(argv[i], "-o")==0){
            output = argv[i+1];
        }
    }

    if (test.rows() != testp.size()) {
        testp = (Eigen::ArrayXd::Random(test.rows()) > 0).cast<int>();
    }

    cout << " >> Start Local Search <<" << endl;

    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto res = localSearch(test, testp, n);
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    cout << "Finished local search in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " miliseconds" << endl;

    saveResults(res, output);
    return 0;
}