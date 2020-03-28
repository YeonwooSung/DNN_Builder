/*
 * variable.cpp
 *
 */
#include <iostream>
#include <chrono>

#include "variable.h"
#include "function.h"

using namespace std;


int count_function = 0;
int count_variable = 0;

map<Variable *, bool> variable_pool;

/**
 * Construct new variable.
 * @param {rows} # of rows of the data
 * @param {cols} # of cols of the data
 * @return The generated variable.
 */
Variable *variable_construct(int rows, int cols){
    count_variable += 1;

    for (auto itr = variable_pool.begin(); itr != variable_pool.end(); ++itr) {
        // reconstruct the variables whose flag value is false.
        if (!itr->second){
            Variable *v = (Variable *)itr->first;
            if (v->data.rows == rows && v->data.cols == cols){
                v->zeros();
                v->creator = NULL;
                variable_pool[v] = true;

                return v;
            }
        }
    }

    // allocate memory for the Variable.
    Variable *r = new Variable(rows, cols);
    // set the flag value as true, so that the program does not re-allocate values to this variable.
    variable_pool[r] = true;

    return r;
}

/**
 * Destory the given variable.
 * If the variable pool has enough space, the program will not free the allocated memory, and mark the
 * variable as not-initialised in the variable pool.
 * However, if the size of the variable pool is greater than the limit, then this function will free
 * the memory that is allocated for the given variable.
 *
 * @param {ptr} The pointer that points the target variable instance.
 */
void variable_destroy(Variable *ptr){
    count_variable -= 1;
    variable_pool[ptr] = false;

    // if the variable pool size is greater than the limit, than erase the target variable.
    if (variable_pool.size() > 4000){
        // erase the variable from the pool
        variable_pool.erase(ptr);
        // free the allocated memory
        delete ptr;
    }
}




//global variable for id
int gVariableId = 0;

/**
 * Returns the gVariableId to set the id of the variable.
 * Before returning the id number, this function updates the 
 * gVariableId value by increasing the value with 1.
 */
int allocateVarId() {
    int id = gVariableId;
    gVariableId += 1;
    return id;
}

// Variable class //////////////////////////////////////////////////////
Variable::Variable(){
    this->init();
    this->id = allocateVarId();
}

Variable::Variable(const Variable &a) {
    this->init();
    this->id = allocateVarId();

    data = a.data;
    grad = a.grad;
    data_sparse = a.data_sparse;
    seed = a.seed;
    creator = a.creator;

    this->isGetGrad = a.isGetGrad;
    this->isSparse = a.isSparse;
}

Variable::Variable(int rows, int cols) {
    this->init();
    this->id = allocateVarId();

    data = cuMat(rows, cols);
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;
}

Variable::Variable(int rows, int cols, bool is_get_grad){
    this->init();
    this->isGetGrad = is_get_grad;
    this->id = allocateVarId();

    //TODO if rows < 0  ->  use the absolute value (multiply -1)
    if (rows < 00) {
        rows = -rows;
    }

    //TODO if cols < 0  ->  use the absolute value (multiply -1)
    if (cols < 0) {
        cols = -cols;
    }

    data = cuMat(rows, cols);
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;
}

Variable::Variable(cuMat &input) {
    this->init();
    this->id = allocateVarId();

    data = input;
    grad = cuMat(input.rows, input.cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;
}

Variable::Variable(Function *f, int rows, int cols) {
    this->init();
    this->id = allocateVarId();

    data = cuMat(rows, cols);
    grad = cuMat(rows, cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = f;
}

Variable::Variable(Function *f, cuMat &input) {
    this->init();
    this->id = allocateVarId();

    data = input;
    grad = cuMat(input.rows, input.cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = f;
}

Variable::Variable(vector<float> &ids, int nums){
    this->init();
    this->id = allocateVarId();

    data_sparse = cuMatSparse(ids, nums);
    grad = cuMat(data_sparse.rows, data_sparse.cols);

    seed = cuMat(grad.rows, grad.cols);
    seed.ones();

    creator = NULL;

    this->isGetGrad = false;
    this->isSparse = true;
}


Variable::~Variable() {
    //TODO free all allocated memory
}



/**
 * Define the operator = for cloning.
 */
Variable &Variable::operator=(const Variable &a) {
    this->init();
    this->id = allocateVarId();

    data = a.data;
    grad = a.grad;

    seed = a.seed;

    creator = a.creator;


    this->isGetGrad = a.isGetGrad;
    this->isSparse = a.isSparse;

    return *this;
}


void Variable::setCreator(Function *f) {
    this->creator = f;
}

void Variable::backward() {
    this->grad = seed;
    this->backward(this);
}

void Variable::backward(Variable *v) {
    if (v == NULL) {
        return;
    }

    if (v->creator != NULL) {

        if (v->last_opt != NULL && v->opt == *v->last_opt){
            *v->is_last_backward = true;
        }

        if (v->forward_count > 0) v->forward_count--;

        if (v->is_last_backward != NULL && *v->is_last_backward == false) return;

        if (v->forward_count != 0) return;

        v->creator->backward(v->grad);

        for (int i = 0; i< v->creator->inputs.size(); i++) {
            PVariable nv = v->creator->inputs[i];

            if (nv->isGetGrad) {
                this->backward(nv.get());
            }
        }

    } else{
        //TODO what if v->creator == NULL ????
    }
}


/**
 * Zero out all gradients of current variable and all connected variables.
 */
void Variable::zero_grads() {
    this->zero_grads(this);
}

/**
 * Zero out gradients of target variables and all variables that are connected with the target variable.
 *
 * @param {v} Pointer that points the target variable.
 */
void Variable::zero_grads(Variable *v) {
    if (v == NULL)
        return;

    v->grad.mul(0, v->grad);
    v->forward_count = 0;

    if (v->creator != NULL) {
        for (int i = 0; i < v->creator->inputs.size(); i++) {
            PVariable nv = v->creator->inputs[i];
            this->zero_grads(nv.get());
        }
    }
}


/**
 * Create variable that is filled with 1s.
 */
void Variable::ones() {
    data.ones();
    grad.mul(0, grad);

}

/**
 * Create variable that is filled with 0s.
 */
void Variable::zeros() {
    data.mul(0, data);
    grad.mul(0, grad);
    forward_count = 0;
    last_opt = NULL;
    is_last_backward = NULL;
    this->creator = NULL;
}

/**
 * Unchain the variable with the function.
 */
void Variable::unchain(){
    this->creator = NULL;
}

/**
 * Zero out the gradients.
 */
void Variable::zero_grad(){
    grad.mul(0, grad);
}

/**
 * Fill the variable with random values.
 */
void Variable::randoms(float m, float a) {
    random_device rd;
    mt19937 mt(rd());
    normal_distribution<float> initd1(m, a);

    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols; j++) {
            data.memSetHost(i, j, initd1(mt));
        }
    }
    data.memHostToDevice();
}


/**
 * Generates binomial distribution.
 */
void Variable::binominal_randoms(float ratio){
    random_device rd;
    mt19937 mt(rd());
    uniform_real_distribution<float> initd1(0., 1.);

    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.cols; j++) {
            float h = 1.0;
            if (initd1(mt) < ratio) h = 0.0;
            data.memSetHost(i, j, h);
        }
    }

    data.memHostToDevice();
}

/**
 * Gets the value of the variable.
 */
float Variable::val(){
    return data(0, 0);
}
