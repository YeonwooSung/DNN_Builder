#ifndef _VARIABLE_
#define _VARIABLE_

#include <list>
#include <random>
#include <memory>
#include <boost/intrusive_ptr.hpp>

#include <matrix/cuMat.h>
#include "cuMatSparse.h"

using namespace std;



class Function;


class Variable {

private:
    friend class boost::serialization::access;
    template<class Archive> void serialize(Archive & ar, const unsigned int version) {
        ar & id;
        ar & data;
        ar & grad;
        ar & seed;
        ar & isGetGrad;
    }

    void init() {
        this->id = 0;
        this->opt = 0;
        this->last_opt = NULL;
        this->is_last_backward = NULL;
        this->forward_count = 0;
        this->creator = NULL;
        this->grad_num = -999;
        this->isGetGrad = true;
        this->isSparse = false;
    }

public:

    int id;
    int opt;
    int *last_opt;
    bool *is_last_backward;

    int forward_count;

    Function *creator = NULL;

    string name;

    cuMat data;
    cuMatSparse data_sparse;
    cuMat grad;
    cuMat seed;

    int grad_num;
    bool isGetGrad;
    bool isSparse;


    // constructors
    Variable();
    Variable(const Variable &a);
    Variable(int rows, int cols);
    Variable(int rows, int cols, bool is_get_grad);
    Variable(Function *f, int rows, int cols);
    Variable(cuMat &input);
    Variable(Function *f, cuMat &input);
    Variable(vector<float> &ids, int nums);

    ~Variable();

    void setCreator(Function *f);

    Variable &operator=(const Variable &a);

    Variable sin();
    Variable log();

    void backward();
    void backward(Variable *v);


    void zero_grads();
    void zero_grads(Variable *v);
    /*
    void truncate();
    void truncate(Variable *v);
        */

    void ones();
    void zeros();
    void unchain();
    void zero_grad();
    void randoms(float m, float a);
    void binominal_randoms(float ratio);
    float val();
};


using PVariable = shared_ptr<Variable>;


Variable *variable_construct(int rows, int cols);
void variable_destroy(Variable *ptr);


#endif
