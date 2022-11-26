#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
# define NUM_THREADS 2

struct timeval start_t , end_t ;

struct thread_param {  
SimpleRNN *rnn;
DerivedSimpleRNN *drnn;
int start;
int end;
float loss;
};

Data *data ;


struct thread_param threads_params[NUM_THREADS];

void *ThreadTrain (void *params) { // Code du thread
struct thread_param *mes_param ;
mes_param = ( struct thread_param *) params ;
for (int i = mes_param->start; i < mes_param->end; i++)
{
    forward(mes_param->rnn, data->X[i], data->xcol , data->embedding);
    backforward(mes_param->rnn, data->xcol, data->Y[i], data->X[i], data->embedding, 
    mes_param->drnn);
	mes_param->loss = mes_param->loss + binary_loss_entropy(data->Y[i], mes_param->rnn->y);
    // acc = accuracy(acc , data->Y[i], rnn->y);
}
pthread_exit (NULL) ;
}

int main()
{
    // srand(time(NULL));


    int size = 4000;
    double totaltime;
    // clock_t start_t, end_t ;
    data = malloc(sizeof(Data));
    get_data(data, 2);
    int n = size/NUM_THREADS;
    int start , end;
    start = 0 ; end = n-1 ;


    int input = 128 , hidden = 64 , output = 2;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr ;
    void *status;
    int r;

    printf("\n----Thread create phase start---- \n");
    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    // start_t = clock();
    gettimeofday(&start_t, NULL);
    printf("\n %d \n", n);

    for (int i = 0; i < 3; i++)
    {

    float Loss = 0.0 ;
    printf("\n epoch %d \n", (i+1));
        
    for ( int i=0; i < NUM_THREADS ; i ++) {

        threads_params[i].rnn = malloc(sizeof(SimpleRNN));
        initialize_rnn(threads_params[i].rnn, input, hidden, output);
        threads_params[i].drnn = malloc(sizeof(DerivedSimpleRNN));
        initialize_rnn_derived(threads_params[i].rnn , threads_params[i].drnn);
        threads_params[i].loss = 0.0;
        threads_params[i].start = start;
        threads_params[i].end = end;


        r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
        if (r) {
            printf("ERROR; pthread_create() return code : %d\n", r);
            exit(-1);
        }
        printf("Thread %d has starded \n", i);

        start = end + 1;
        end = end + n;

    }
    printf("----Thread create phase end----\n");


    /* Free attribute and wait for the other threads */
    pthread_attr_destroy(&attr);

    for(int t=0; t<NUM_THREADS; t++) {
        r = pthread_join(threads[t], &status);
        if (r) {
            printf("ERROR; return code from pthread_join() is %d\n", r);
            exit(-1);
        }
        Loss = Loss + threads_params[t].loss ;
        printf("Main: completed join with thread %d an loss = %f\n",t, (threads_params[t].loss)/n);
    }
    
    printf("--> Loss : %f  \n" , Loss/size);    
 
        
    }

    gettimeofday(&end_t, NULL);
    totaltime = (((end_t.tv_usec - start_t.tv_usec) / 1.0e6 + end_t.tv_sec - start_t.tv_sec) * 1000) / 1000;
    printf("\nTRAINING PHASE END IN %lf s\n" , totaltime);
    
    
    return 0 ;
}
