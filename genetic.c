// genetic.c - Example training of ann.h ANN, with genetic algorithm

#define GENETIC_IMPLEMENTATION
#include "genetic.h"

#define ANN_IMPLEMENTATION
#include "ann.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#define POPULATION_N 100

double network_fitness( void *x, void const *c )
{
    ann_t *ann = ( ann_t * ) x;
    fp_t const *i = *( ( fp_t ** ) c );
    fp_t const *t = *( ( fp_t ** ) c  + 1);
    fp_t o[ann->layer_neuron_n[ann->layer_n - 1]];
    
    fp_t e = 0;
	for( uint_t b = 0; b < 4; b++ )
	{
		ann_propagation_forward( ann, ( fp_t const * const ) &i[2 * b], &o[b] );
        e += ann_error_total( &o[b], &t[b], ann->layer_neuron_n[ann->layer_n - 1] );
    }
    
    return -e;
}

void network_crossover( void *y, void *x0, void *x1, double rate )
{
    ann_t *ay = ( ann_t * ) y;
    ann_t *ax0 = ( ann_t * ) x0;
    ann_t *ax1 = ( ann_t * ) x1;
    
    for( uint_t i = 0; i < ay->weight_bias_n; i++ )
    {
        if( ( double ) rand() / RAND_MAX < rate )
        {
            ay->weight_bias[i] = ( rand() % 2 ) ? ax0->weight_bias[i] : ax1->weight_bias[i];
        }
    }
}

void network_mutate( void *x, double rate )
{
    ann_t *ann = ( ann_t * ) x;
    
    for( uint_t i = 0; i < ann->weight_bias_n; i++ )
    {
        if( ( double ) rand() / RAND_MAX < rate )
        {
            ann->weight_bias[i] += ( double ) rand() / RAND_MAX / 10 - 0.5;
        }
    }
}

int main(void)
{
    srand( time( 0 ) );
   
	fp_t input[12] = { 0, 0, 0, 1, 1, 0, 1, 1 };
	fp_t output;
	fp_t target[4] = { 0, 1, 0.4, 0 };
   
    ann_t **networks = malloc( sizeof( ann_t * ) * POPULATION_N );

    for( uint_t i = 0; i < POPULATION_N; i++ )
    {
        networks[i] = ann_init( 3, ( uint_t[] ){ 2, 2, 1 } );
        ann_set_activation( networks[i], TANH, IDENTITY );
	    ann_random( networks[i] );
    }

    fp_t e = 1;
    while( e > 0.001 )
    {        
        genetic_generation( 
            networks, 
            ( void * ) ( fp_t *[2] ){ input, target },
            POPULATION_N, 
            0.1, 
            0.1,
            0.1, 
            network_fitness,
            network_crossover, 
            network_mutate
        );
        
        for( int j = 0; j < 4; j++ )
        {
        	ann_propagation_forward( networks[0], ( fp_t const * const ) &input[2 * j], &output );
            e += ann_error_total( &output, &target[j], networks[0]->layer_neuron_n[networks[0]->layer_n - 1] );
        }
        
        e /= 4;
    }
    
    for( int i = 0; i < 1; i++ )
    {
        for( int j = 0; j < 4; j++ )
        {
        
            //ann_print_weight_bias( networks[i] );
        	ann_propagation_forward( networks[i], ( fp_t const * const ) &input[2 * j], &output );
            ann_print_neuron( networks[i], &input[2 * j], &output );
        }
    }
}
