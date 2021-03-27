//
// Created by chenliguo on 10/23/18.
//

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thread>
#include "npy.hpp"
#include "rapidjson/document.h"
#include "Matrix_Factorization.h"
#include <boost/algorithm/string.hpp>

using namespace std;

Matrix_Factorization::Matrix_Factorization(const char *configure_path) {
    // load onf
    ios_base::sync_with_stdio(false);
    this->user_count = 0;
    this->items_count = 0;
    this->tokens_count = 0;
    rapidjson::Document config;
    ifstream config_file(configure_path);
    string str;
    string json_str;
    while (getline(config_file, str))
        json_str += str;
    config_file.close();
    config.Parse(json_str.c_str());
    if (!config.IsObject()) {
        cerr << configure_path << " is not valid json." << endl;
        exit(-1);
    }
    if (!config.HasMember("train_input_path")) {
        cerr << configure_path << "does not contains 'train_input_path'." << endl;
        exit(-2);
    }
    if (!config["train_input_path"].IsString()) {
        cerr << configure_path << ": train_input_path is not string." << endl;
        exit(-3);
    }
    this->train_input_path = config["train_input_path"].GetString();
    this->embedding_size = config["embedding_size"].GetUint();
    this->threads = config["threads"].GetUint();
    this->epoch = config["epoch"].GetUint();
    this->learning_rate = config["learning_rate"].GetDouble();
    this->reg_rate = config["reg_rate"].GetDouble();
    this->negative_samples = config["negative_samples"].GetUint();
    this->uid_path = config["uid_path"].GetString();
    this->iid_path = config["iid_path"].GetString();
    this->item_embeddings_path = config["item_embedding_path"].GetString();
    this->user_embeddings_path = config["user_embedding_path"].GetString();

    // init
    this->users_dict = new unordered_map<string, unsigned long>();
    this->items_dict = new unordered_map<string, unsigned long>();
    this->users = new vector<string>();
    this->items = new vector<string>();
}

Matrix_Factorization::~Matrix_Factorization() {
    delete this->users_dict;
    delete this->items_dict;
    delete this->users;
    delete this->items;
    delete tokens;
    delete user_embeddings;
    delete item_embeddings;
}

bool Matrix_Factorization::load_data() {
    // read fast
    ios_base::sync_with_stdio(false);
    ifstream train_input(this->train_input_path);
    string str;
    vector<string> sections;
    // load users, items
    while (getline(train_input, str)) {
        boost::trim(str);
        boost::split(sections, str, boost::is_any_of(" "));
        if (sections.size() != 2)
            continue;
        if (users_dict->find(sections[0]) == users_dict->end()) {
            users_dict->insert({sections[0], users->size()});
            users->push_back(sections[0]);
        }
        if (items_dict->find(sections[1]) == items_dict->end()) {
            items_dict->insert({sections[1], items->size()});
            items->push_back(sections[1]);
        }
        this->tokens_count++;
    }
    train_input.clear();
    train_input.seekg(0);
    // init user embeddings and item embeddings
    user_count = users->size();
    items_count = items->size();
    user_embeddings = new double[user_count * embedding_size];
    item_embeddings = new double[items_count * embedding_size];
    for (unsigned int i = 0; i < user_count; i++) {
        for (unsigned int j = 0; j < embedding_size; j++) {
            user_embeddings[i * embedding_size + j] = double(rand()) / (double(RAND_MAX) + 1.0) - 0.5;
        }
    }
    for (unsigned int i = 0; i < items_count; i++) {
        for (unsigned int j = 0; j < embedding_size; j++) {
            item_embeddings[i * embedding_size + j] = double(rand()) / (double(RAND_MAX) + 1.0) - 0.5;
        }
    }
    // load user clicks
    this->tokens = new unsigned long[this->tokens_count * 2];
    int i = 0;
    while (getline(train_input, str)) {
        boost::trim(str);
        boost::split(sections, str, boost::is_any_of("\t"));
        if (sections.size() != 2)
            continue;
        this->tokens[i * 2] = users_dict->find(sections[0])->second;
        this->tokens[i * 2 + 1] = items_dict->find(sections[1])->second;
        i += 1;
    }
    train_input.close();
    return true;
}

void static update(Matrix_Factorization *t, unsigned long uid, unsigned long iid, int y, double &l) {
    double s = 0.0, e;
    for (int i = 0; i < t->embedding_size; i++) {
        s += t->user_embeddings[uid * t->embedding_size + i] *
             t->item_embeddings[iid * t->embedding_size + i];
    }
    s = 1.0 / (1.0 + exp(-s));
    e = y - s;
    for (int i = 0; i < t->embedding_size; i++) {
        t->user_embeddings[uid * t->embedding_size + i] +=
                (t->learning_rate * e * t->item_embeddings[iid * t->embedding_size + i] -
                 t->reg_rate * t->user_embeddings[uid * t->embedding_size + i]);
        t->item_embeddings[iid * t->embedding_size + i] +=
                (t->learning_rate * e * t->user_embeddings[uid * t->embedding_size + i] -
                 t->reg_rate * t->item_embeddings[iid * t->embedding_size + i]);
    }
    l -= y == 1.0 ? log(s) : log(1.0 - s);
}

bool static train_part(Matrix_Factorization *t, size_t start, size_t stop, int pid) {
    unsigned long uid = 0, iid = 0;
    double ploss = 0.0, nloss = 0.0;
    unsigned int seed = time(nullptr);
    for (int epoch_cu = 0; epoch_cu < t->epoch; epoch_cu++) {
        for (size_t tid = start; tid < stop; tid++) {
            uid = t->tokens[tid * 2];
            iid = t->tokens[tid * 2 + 1];
            update(t, uid, iid, 1, ploss);
            for (size_t j = 0; j < t->negative_samples; j++) {
                // random sampling from items
                iid = uint(rand_r(&seed) * 1.0 / RAND_MAX * t->items_count);
                update(t, uid, iid, 0, nloss);
            }
        }
        ploss /= (stop - start);
        nloss /= (stop - start) * t->negative_samples;
        if (pid == 0)
            printf("epoch %3d:    ploss:%.4f    nloss:%.4f\n", epoch_cu, ploss, nloss);
        ploss = 0.0;
        nloss = 0.0;
    }
    return true;
}

bool Matrix_Factorization::train() {
    size_t tokens_single_thread = tokens_count / threads;
    size_t left = tokens_count % threads;

    thread trainers[this->threads];
    for (int i = 0; i < threads; i++) {
        trainers[i] = thread(train_part, this, i * tokens_single_thread,
                             (i + 1) * tokens_single_thread + (i == threads - 1 ? left : 0), i);
    }
    for (int i = 0; i < threads; i++) {
        trainers[i].join();
    }
    return true;
}

bool Matrix_Factorization::save() {
    // save users
    std::ofstream uid_op(uid_path);
    for (int i = 0; i < user_count; i++) {
        uid_op << (*users)[i] << "\n";
    }
    uid_op.close();
    // save items
    std::ofstream iid_op(iid_path);
    for (int i = 0; i < items_count; i++) {
        iid_op << (*items)[i] << "\n";
    }
    iid_op.close();
    // save user emb
    const unsigned long ushape[] = {user_count, embedding_size};
    vector<double> user_emb{user_embeddings, user_embeddings + user_count * embedding_size};
    npy::SaveArrayAsNumpy(user_embeddings_path, false, 2, ushape, user_emb);
    // save item emb
    const unsigned long ishape[] = {items_count, embedding_size};
    vector<double> item_emb{item_embeddings, item_embeddings + items_count * embedding_size};
    npy::SaveArrayAsNumpy(item_embeddings_path, false, 2, ishape, item_emb);
    return true;
}
