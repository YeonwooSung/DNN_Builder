/*
 * cuMatSparse.h
 */

#ifndef CUMATSPARSE_H_
#define CUMATSPARSE_H_

#include<iostream>
#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<cusparse_v2.h>
#include<thrust/device_vector.h>

#include "cuMat.h"

class cuMatSparse {

private:

    void init() {
        this->rows = 0;
        this->cols = 0;
        this->csrVal = NULL;
        this->csrRowPtr = NULL;
        this->csrColInd = NULL;
        this->csrValDevice = NULL;
        this->csrRowPtrDevice = NULL;
        this->csrColIndDevice = NULL;
    }


public:

    int rows;
    int cols;

    cusparseHandle_t cuHandle;
    cusparseMatDescr_t descr;

    float *csrVal = NULL;
    int *csrRowPtr = NULL;
    int *csrColInd = NULL;

    float *csrValDevice;
    int *csrRowPtrDevice;
    int *csrColIndDevice;


    int numVals = 0;

    cuMat rt, bt;


    cuMatSparse() {
        this->init();

        cusparseCreate(&cuHandle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }

    cuMatSparse(int rows, int cols, int numberOfVals) {
        this->init();

        cout << "cuMatSparse(int rows, int numberOfVals)" << endl;

        cusparseCreate(&cuHandle);
        cusparseCreateMatDescr(&descr);
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

        new_matrix(rows, cols, numberOfVals);
    }

    cuMatSparse(vector<float> &ids, int col_nums) : cuMatSparse(){
        this->init();
        embed(ids, col_nums);
    }


    ~cuMatSparse(){
        cusparseDestroyMatDescr(descr);
        cusparseDestroy(cuHandle);

        free(csrVal);
        free(csrRowPtr);
        free(csrColInd);
        cudaFree(csrValDevice);
        cudaFree(csrRowPtrDevice);
        cudaFree(csrColIndDevice);


    }

    void new_matrix(int rows, int cols, int numberOfVals){
        this->rows = rows;
        this->cols = cols;
        this->numVals = numberOfVals;

        cudaError_t error = cudaMalloc((void**) &csrValDevice, numberOfVals * sizeof(*csrValDevice));
        error = cudaMalloc((void**) &csrRowPtrDevice, (rows+1) * sizeof(*csrRowPtrDevice));
        error = cudaMalloc((void**) &csrColIndDevice, numberOfVals * sizeof(*csrColIndDevice));

        cudaMemset(csrValDevice, 0x00, numberOfVals * sizeof(*csrValDevice));
        cudaMemset(csrRowPtrDevice, 0x00, (rows+1)  * sizeof(*csrRowPtrDevice));
        cudaMemset(csrColIndDevice, 0x00, numberOfVals * sizeof(*csrColIndDevice));
    }

    cuMatSparse &operator=(const cuMatSparse &a) {
        new_matrix(a.rows, a.cols, a.numVals);

        cudaError_t error = cudaMemcpy(csrValDevice, a.csrValDevice, a.numVals * sizeof(*csrValDevice), cudaMemcpyDeviceToDevice);
        error = cudaMemcpy(csrRowPtrDevice, a.csrRowPtrDevice, (a.rows+1) * sizeof(*csrRowPtrDevice), cudaMemcpyDeviceToDevice);
        error = cudaMemcpy(csrColIndDevice, a.csrColIndDevice, a.numVals * sizeof(*csrColIndDevice), cudaMemcpyDeviceToDevice);

        return *this;
    }

    void zeros(){
        cudaMemset(csrValDevice, 0x00, numVals * sizeof(*csrValDevice));
        cudaMemset(csrRowPtrDevice, 0x00, (rows+1)  * sizeof(*csrRowPtrDevice));
        cudaMemset(csrColIndDevice, 0x00, numVals * sizeof(*csrColIndDevice));
    }

    void memSetHost(float *v, int *r, int *c) {
        cudaError_t error = cudaMemcpy(csrValDevice, v, numVals * sizeof(*csrValDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            printf("memSetHost cudaMemcpy error: csrValDevice\n");
        error = cudaMemcpy(csrRowPtrDevice, r, (rows+1) * sizeof(*csrRowPtrDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            printf("memSetHost cudaMemcpy error: csrRowPtrDevice\n");
        error = cudaMemcpy(csrColIndDevice, c, numVals * sizeof(*csrColIndDevice), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            printf("memSetHost cudaMemcpy error: csrColIndDevice\n");
    }

    //column majar format
    void embed(vector<float> &ids, int col_nums){
        this->rows = ids.size();
        this->cols = col_nums;

        int num_vals = rows;
        numVals = num_vals;


        csrVal = (float *)malloc(num_vals * sizeof(*csrVal));
        csrRowPtr = (int *)malloc((rows+1) * sizeof(*csrRowPtr));
        csrColInd = (int *)malloc(num_vals * sizeof(*csrColInd));

        cudaError_t error = cudaMalloc((void**) &csrValDevice, num_vals * sizeof(*csrValDevice));
        error = cudaMalloc((void**) &csrRowPtrDevice, (rows+1) * sizeof(*csrRowPtrDevice));
        error = cudaMalloc((void**) &csrColIndDevice, num_vals * sizeof(*csrColIndDevice));


        memset(csrRowPtr, 0x00, (rows+1) * sizeof(*csrRowPtr));
        csrRowPtr[0] = 0;

        for(int i=0; i<rows; i++){
            csrVal[i] = 1.; //value is 1
            csrColInd[i] = ids[i];
            csrRowPtr[i+1] = csrRowPtr[i] + 1; //only a element per row
        }

        memSetHost(csrVal, csrRowPtr, csrColInd);
    }


    void s_s_dot(cuMatSparse &b, cuMatSparse &c){

        cusparseStatus_t status =
                cusparseScsrgemm(cuHandle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 rows,
                                 b.cols,
                                 cols,
                                 descr,
                                 numVals,
                                 csrValDevice,
                                 csrRowPtrDevice,
                                 csrColIndDevice,
                                 b.descr,
                                 b.numVals,
                                 b.csrValDevice,
                                 b.csrRowPtrDevice,
                                 b.csrColIndDevice,
                                 c.descr,
                                 c.csrValDevice,
                                 c.csrRowPtrDevice,
                                 c.csrColIndDevice );

        if (status != CUSPARSE_STATUS_SUCCESS)
            cout << "ERROR cuMatSparse::s_s_dot cusparseXcsrgeamNnz" << endl;
        cudaThreadSynchronize();
    }

    void s_d_dot(cuMat &b, cuMat &c){

        float alpha = 1.;
        float beta = 0.;
        cusparseStatus_t status = cusparseScsrmm(cuHandle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
            rows,
            b.cols,
            cols,
            numVals,
            &alpha,
            descr,
            csrValDevice,
            csrRowPtrDevice,
            csrColIndDevice,
            b.mDevice,
            b.rows,
            &beta,
            c.mDevice,
            c.rows);

        if (status != CUSPARSE_STATUS_SUCCESS) {
            cout << "ERROR cuMatSparse::s_d_dot cusparseScsrmm" << endl;
            cout << "a rows:" << rows << " cols:" << cols << endl;
            cout << "b rows:" << b.rows << " cols:" << b.cols << endl;
            cout << "c rows:" << c.rows << " cols:" << c.cols << endl;
            switch(status) {
            case CUSPARSE_STATUS_NOT_INITIALIZED:
                cout << "CUSPARSE_STATUS_NOT_INITIALIZED" << endl;
                break;
            case CUSPARSE_STATUS_ALLOC_FAILED:
                cout << "CUSPARSE_STATUS_ALLOC_FAILED" << endl;
                break;
            case CUSPARSE_STATUS_INVALID_VALUE:
                cout << "CUSPARSE_STATUS_INVALID_VALUE" << endl;
                break;
            case CUSPARSE_STATUS_ARCH_MISMATCH:
                cout << "CUSPARSE_STATUS_ARCH_MISMATCH" << endl;
                break;
            case CUSPARSE_STATUS_EXECUTION_FAILED:
                cout << "CUSPARSE_STATUS_EXECUTION_FAILED" << endl;
                break;
            case CUSPARSE_STATUS_INTERNAL_ERROR:
                cout << "CUSPARSE_STATUS_INTERNAL_ERROR" << endl;
                break;
            case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                cout << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED" << endl;
                break;
            }
        }

        cudaThreadSynchronize();
    }


    void d_s_dot(cuMat &b, cuMat &r){
        cuMatSparse t = this->transpose(); //waste time here

        if (rt.rows == 0)
            rt = r.transpose();
        if (bt.rows == 0)
            bt = b.transpose();
        b.transpose(bt);
        t.s_d_dot(bt, rt);
        rt.transpose(r);
    }


    void transpose(cuMatSparse &r){
        cusparseStatus_t status = cusparseScsr2csc(cuHandle, rows, cols, numVals,
                         csrValDevice, csrRowPtrDevice,
                         csrColIndDevice, r.csrValDevice,
                         r.csrColIndDevice, r.csrRowPtrDevice,
                         CUSPARSE_ACTION_NUMERIC,
                         CUSPARSE_INDEX_BASE_ZERO);

        if (status != CUSPARSE_STATUS_SUCCESS)
            cout << "transpose error" << endl;
        cudaThreadSynchronize();
    }


    cuMatSparse transpose() {
        cuMatSparse r(cols, rows, numVals);
        transpose(r);
        return r;
    }

    cuMat toDense(){
        cuMat r(rows, cols);

        cusparseStatus_t status = cusparseScsr2dense(cuHandle,
                                    r.rows,
                                    r.cols,
                                    descr,
                                    csrValDevice,
                                    csrRowPtrDevice,
                                    csrColIndDevice,
                                    r.mDevice,
                                    rows);

        if (status != CUSPARSE_STATUS_SUCCESS)
                    cout << "toDense error" << endl;
        cudaThreadSynchronize();

        return r;
    }

    cuMatSparse toSparse(cuMat &a, int numVals){

        cuMatSparse r(a.rows, a.cols, a.rows);

        int *nnzPerRowColumn;
        cudaMalloc((void **)&nnzPerRowColumn, sizeof(int) * r.rows);
        int nnzTotalDevHostPtr = numVals;
        cusparseStatus_t status = cusparseSnnz(r.cuHandle, CUSPARSE_DIRECTION_ROW, r.rows, r.cols, r.descr, 
            a.mDevice, r.rows, nnzPerRowColumn, &nnzTotalDevHostPtr);

        if (status != CUSPARSE_STATUS_SUCCESS)
            cout << "toSparse cusparseSnnz error" << endl;

        cudaThreadSynchronize();


        status = cusparseSdense2csr(r.cuHandle, r.rows, r.cols,
                        r.descr,
                        a.mDevice,
                        r.rows, nnzPerRowColumn,
                        r.csrValDevice,
                        r.csrRowPtrDevice, r.csrColIndDevice);

        if (status != CUSPARSE_STATUS_SUCCESS)
            cout << "toSparse cusparseSdense2csr error" << endl;
        cudaThreadSynchronize();

        return r;
    }
};


#endif /* CUMATSPARSE_H_ */
