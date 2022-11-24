#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <time.h>
#include <string.h>
#include <pthread.h>

int main()
{

    // float label[4] = {0,1,0.3,0.56};
    // double y_pred[2] = {0.60,0.40};

    // double loss = binary_loss_entropy(label, y_pred, 2);

    // printf("%lf", loss);

    // Data *data = malloc(sizeof(Data));
    // get_data(data);

//   printf("\n ***************** IMPORT PHASE START *****************\n");
//   Data *data = malloc(sizeof(Data));
//   get_data(data, 2);

   int n = 2;

   float mean = 1/(float)n ;

    printf("%f \n", mean);



 
    
    return 0 ;
}
