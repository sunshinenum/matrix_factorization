//
// Created by chenliguo on 10/23/18.
//
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

#ifndef MATRIX_FACTORIZATION_MATRIX_FACTORIZATION_H
#define MATRIX_FACTORIZATION_MATRIX_FACTORIZATION_H

class Matrix_Factorization
{
public:
    // conf
    unsigned int embedding_size;
    unsigned int threads;
    unsigned int epoch;
    double learning_rate;
    double reg_rate;
    unsigned long negative_samples;
    string train_input_path;
    string uid_path;
    string iid_path;
    string item_embeddings_path;
    string user_embeddings_path;

    // train
    unordered_map<string, unsigned long> *users_dict;
    unordered_map<string, unsigned long> *items_dict;
    vector<string> *users;
    vector<string> *items;
    unsigned long *tokens{};
    double *user_embeddings{};
    double *item_embeddings{};
    unsigned long user_count;
    unsigned long items_count;
    unsigned long tokens_count;

    // functions
    Matrix_Factorization(const char *configure_path);
    ~Matrix_Factorization();
    bool load_data();
    bool train();
    bool save();
};

#endif //MATRIX_FACTORIZATION_MATRIX_FACTORIZATION_H
