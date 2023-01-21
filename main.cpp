#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <chrono>

using namespace std;

typedef Eigen::MatrixXi Graph;
typedef Eigen::VectorXi Perm;

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

    cout << "Reading file -> ";

    string delimiter = " ";

    size_t token = stoi(line.substr(0, line.find(delimiter)));
    line.erase(0, line.find(delimiter) + delimiter.length());

    Graph graph = Graph::Constant(token, token, 0);

    cout << "Graph with " << token << " vertices and ";

    token = stoi(line.substr(0, line.find(delimiter)));

    cout << token << " edges." << endl;

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

struct ls_output {
    Perm perm;
    int cost;
};

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

            int sub_cause_i = p_m.dot(graph.row(i)) + p_m.dot(graph.col(i));

            new_p(i) ^= 1;

            for (int j=0; j< perm.size(); j++){
                if (p(j)) continue;

                new_p(j) ^= 1;

                new_p_m = ones - new_p;

                cost_p -= p.dot(graph.row(j)) + p.dot(graph.col(j)) + sub_cause_i;

                cost_p += new_p_m.dot(graph.row(j)) +
                          new_p_m.dot(graph.col(j)) +
                          new_p.dot(graph.row(i)) +
                          new_p.dot(graph.col(i));

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

int main(){
    int n = 1000;
    Graph test = (Eigen::ArrayXXd::Random(n, n) > 0).cast<int>();
    Perm testp = (Eigen::ArrayXd::Random(test.rows()) > 0).cast<int>();

    // Graph test = loadGraph("test.txt");
    // Perm testp(6);
    // testp << 1, 1, 1, 0, 0, 0;

    cout << cost(test, testp) << endl;

    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto res = localSearch(test, testp, 1);
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // cout << res.perm.transpose() << "\n" << res.cost << endl;

    cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << endl;
}