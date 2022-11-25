#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
# define NUM_THREADS 4

struct thread_param {  
SimpleRNN *rnn;
DerivedSimpleRNN *drnn;
float loss;
long retour;
};

struct thread_param threads_params[NUM_THREADS];

void *ThreadTrain (void *params) { // Code du thread
struct thread_param *mes_param ;
mes_param = ( struct thread_param *) params ;
sleep(6) ;
mes_param->retour = mes_param->retour + mes_param->retour;
pthread_exit (( void *) mes_param->retour) ;
}

int main()
{

    int input = 128 , hidden = 64 , output = 2;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr ;
    void *status;
    int r;

    printf("----Thread create phase start---- \n");
    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for ( int i=0; i < NUM_THREADS ; i ++) {

        threads_params[i].rnn = malloc(sizeof(SimpleRNN));
        initialize_rnn(threads_params[i].rnn, input, hidden, output);
        threads_params[i].drnn = malloc(sizeof(DerivedSimpleRNN));
        initialize_rnn_derived(threads_params[i].rnn , threads_params[i].drnn);
        threads_params[i].retour = i;

        r = pthread_create (&threads[i] ,&attr ,ThreadTrain ,(void*)&threads_params[i]) ;
        if (r) {
            printf("ERROR; pthread_create() return code : %d\n", r);
            exit(-1);
        }
        printf("Thread %d has starded \n", i);

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
        printf("Main: completed join with thread %d an return %ld\n",t, threads_params[t].retour);
    }




 
    
    return 0 ;
}
