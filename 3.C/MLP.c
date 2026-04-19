#include <stdio.h>

# define HEIGHT 20
# define WIDTH 30

float feed_forward(float inputs[HEIGHT][WIDTH] , float weights[HEIGHT][WIDTH])
{
    float outputs= 0.0;
    for (int i = 0 ; i<HEIGHT ; i++)
    {
        for (int j = 0 ; j<WIDTH ; j++)
        {
            outputs += inputs[i][j] * weights[i][j];
        }
    }
    return outputs;
}






int main(void)
{   
    float inputs[HEIGHT][WIDTH];
    float weights[HEIGHT][WIDTH];
    
    float outputs = feed_forward(inputs,weights);
    printf("outputs = %f\n" , outputs);
    return 0;
}