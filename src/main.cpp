#include <iostream>
#include "Matrix_Factorization.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "\
        {\n\
            \"train_input_path\": \"../data/train_input_last\",\n\
            \"embedding_size\": 50, \n\
            \"epoch\": 50, \n\
            \"threads\": 8,\n\
            \"learning_rate\": 0.01,\n\
            \"reg_rate\": 0.00001,\n\
            \"negative_samples\": 2,\n\
            \"uid_path\": \"../data/o_uids\",\n\
            \"iid_path\": \"../data/o_iids\",\n\
            \"user_embedding_path\": \"../data/o_ue.npy\",\n\
            \"item_embedding_path\": \"../data/o_ie.npy\"\n\
        }\
        " << std::endl;
        std::cerr << "matrix_factorization configure_path" << std::endl;
        return -1;
    }
    Matrix_Factorization mf(argv[1]);
    if (!mf.load_data())
    {
        std::cerr << "load data failed!" << std::endl;
        return -2;
    }
    if (!mf.train())
    {
        std::cerr << "train failed!" << std::endl;
        return -3;
    }
    if (!mf.save())
    {
        std::cerr << "save failed!" << std::endl;
        return -4;
    }
    std::cerr << "Matrix Factorization Done" << std::endl;
    return 0;
}
