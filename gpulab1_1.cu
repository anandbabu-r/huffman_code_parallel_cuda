#include <stdlib.h>
#include <stdio.h>
#include <time.h>
//#include "cutil.h"

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        printf("Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        printf("\nPress ENTER to exit...\n");
        getchar();
        exit(-1);
    }
}
//name of the input file
#define INPUT_FILE_NAME "input.txt"
//name of the compressed file
#define COMPRESSED_FILE_NAME "compressed.txt"
#define COMPRESSED_FILE_NAME_GPU "compressed_gpu.txt"

//name of the uncompressed file
#define DECOMPRESSED_FILE_NAME "decompressed.txt"
//name of the config file
#define  CONFIG_FILE_NAME "config.txt"
//max number of characters
#define MAX_CHAR 256

//#define MAX_CHAR 30
//max lenght of the number which can occur in char_frequency or char_huffman_table)
#define MAX_LENGTH_OF_NUMBER 10
//lenght of the array in shared memory on device
#define SHARED_MEMORY_SIZE 256
//lenght of the array in const memory on device
#define CONST_MEMORY_SIZE 15000 //(MAX_CHAR*(MAX_CHAR-1))

//To fill and pass the file as an array to GPU
#define MAX_FILE_CHARS 50000
#define BLOCK_SIZE 256


struct node {
   int val;
   int weight;
   struct node * right, * left;
};

//keeps frequency of particular characters (index - symbof of the character, value - frequency of the character)
int char_frequency[MAX_CHAR];
//keeps huffman table
int char_huffman_table[MAX_CHAR][MAX_CHAR-1];
//keeps number which tells how many bits were unused in last byte (variable is set after call compress_file())
int last_byte_padding=0;

//for writing gpu output
int last_byte_padding_gpu = 0;


//keeps number of characters in current input file - file has to have less than 2,147,483,647 characters (variable is set after call read_file())
int number_of_char=0 ;

//To fill and pass the file as an array to GPU
unsigned char *h_input=0,*d_input=0;

// To read char_huffman_table at the GPU
int *d_char_huffman_table=0;

int copiedarray2[MAX_CHAR][MAX_CHAR-1];


__device__ int char_huffman_table_gpu[MAX_CHAR][MAX_CHAR-1];

//To write the output from compression in GPU

 //char *compressedfile_array=0;

 bool *compressedfile_array=0;

 bool *finalcompressed_array=0;

 // To keep track of how many characters each block wrote

 int *block_cntr_array=0;
 int *block_cntr_array_check=0;
 int *d_last_byte_padding=0;

 int *finalsize=0;
 int *orig_number_of_char=0;
 int *huffman_check = (int *)malloc((MAX_CHAR)*(MAX_CHAR-1) *sizeof(int));

 bool *d_bool = 0;

 bool *h_bool = 0;



 __global__ void final_compression(int *block_cntr_array,bool *compressedfile_array,bool *finalcompressed_array,int number_of_char)
 //__device__ void final_compression(int *block_cntr_array,bool *compressedfile_array,bool *finalcompressed_array)
{
int index_blocks=blockIdx.x*blockDim.x+threadIdx.x;
int index_file=(blockIdx.x*blockDim.x+threadIdx.x)*255;
int final_index=0;

if(index_blocks < number_of_char)
{
for(int i=0;i<index_blocks;i++)
{
final_index = final_index+ block_cntr_array[i];
}
for(int i=0;i<block_cntr_array[index_blocks];i++)
{
finalcompressed_array[final_index+i]=compressedfile_array[index_file+i];
}

}
}


//__global__ void computearray_size(int* block_cntr_array,int *finalsize,int *orig_number_of_char)
__device__ void computearray_size(int* block_cntr_array,int *finalsize,int *orig_number_of_char)
{
*finalsize = 0;
for(int i=0;i<*orig_number_of_char;i++)
{
(*finalsize)=(*finalsize) + block_cntr_array[i];
}

}



/*__global__ void compress_file_gpu(unsigned char *d_input,char *compressedfile_array,int *char_huffman_table2,int *block_cntr_array,int* d_last_byte_padding)
{
	int write_counter=0,block_counter=0;	//how many bits have been written in specific byte
	 unsigned char input_char;
	unsigned char output_char = 0x0;
	 unsigned char end_of_file = 255;
	unsigned char mask = 0x01; //00000001;
	int index_file=(blockIdx.x*blockDim.x+threadIdx.x)*255;
	int index_blocks=blockIdx.x*blockDim.x+threadIdx.x;
	

	//for(int i=0;i<MAX_CHAR;i++)
	//{
		//int *row = (int*)((char*)char_huffman_table2 + i * pitch);
		//for (int c = 0; c < MAX_CHAR-1; ++c) {
          //   char_huffman_table_gpu[i][c] = row[c];
        //}
	//}

  			input_char = d_input[index_blocks];			
			for(int i = 0 ; i < (MAX_CHAR - 1) ; i++)				
			{
				if(char_huffman_table2[input_char*255+i] == 0)			//detect if current character on particular position has 0 or 1
				{
					output_char = output_char << 1;					//if 0 then shift bits one position to left (last bit after shifting is 0)
					write_counter++;
					block_counter++;
				}
				else if(char_huffman_table2[input_char*255+i] == 1)
				{
					output_char = output_char << 1;					//if 1 then shift bits one position to left...
					output_char = output_char | mask;				//...and last bit change to: 1
					write_counter++;
					block_counter++;
				}
				else //-1
				{
					//if(input_char == end_of_file)					//if EOF is detected then write current result to file
					//{													
						if(write_counter != 0)
						{
							output_char = output_char << (8-write_counter);
							compressedfile_array[index_file]=output_char;
							output_char = 0x0;
						}
						else	//write_counter == 0
						{
							compressedfile_array[index_file]=output_char;
						}
					//}

					break;
				}

				if(write_counter == 8)								//if result achieved 8 (size of char) then write it to compressed_file
				{
					compressedfile_array[index_file]=output_char;
					output_char = 0x0;
					write_counter = 0;
				}
			}
		
		block_cntr_array[index_blocks]=block_counter;
		*d_last_byte_padding = write_counter;							//to decompress file we have to know how many bits in last byte have been written
		//update_config(write_counter); //TODO to zakomentowac przy ostatecznych pomiarach
	
}*/

//__global__ void compress_file_gpu(unsigned char *d_input,bool *compressedfile_array,int *char_huffman_table2,int *block_cntr_array,int* d_last_byte_padding)
__global__ void compress_file_gpu(unsigned char *d_input,bool *compressedfile_array,int *char_huffman_table2,int *block_cntr_array,int* d_last_byte_padding,int *finalsize,int *orig_number_of_char,int number_of_char)
{
	//int write_counter=0,
	int block_counter=0;	//how many bits have been written in specific byte
	 unsigned char input_char;
	//unsigned char output_char = 0x0;
	 //unsigned char end_of_file = 255;
	//unsigned char mask = 0x01; //00000001;
	int index_file=(blockIdx.x*blockDim.x+threadIdx.x)*255;
	int index_blocks=blockIdx.x*blockDim.x+threadIdx.x;
	
	if(index_blocks < number_of_char) 
	{
	//for(int i=0;i<MAX_CHAR;i++)
	//{
		//int *row = (int*)((char*)char_huffman_table2 + i * pitch);
		//for (int c = 0; c < MAX_CHAR-1; ++c) {
          //   char_huffman_table_gpu[i][c] = row[c];
        //}
	//}

  			input_char = d_input[index_blocks];			
			for(int i = 0 ; i < (MAX_CHAR - 1) ; i++)				
			{
				if(char_huffman_table2[input_char*255+i] == 0)			//detect if current character on particular position has 0 or 1
				{
					//output_char = output_char << 1;					//if 0 then shift bits one position to left (last bit after shifting is 0)
					compressedfile_array[index_file+i] = false;
					//write_counter++;
					block_counter++;
				}
				else if(char_huffman_table2[input_char*255+i] == 1)
				{
					//output_char = output_char << 1;					//if 1 then shift bits one position to left...
					//output_char = output_char | mask;				//...and last bit change to: 1
					//write_counter++;
					compressedfile_array[index_file+i] = true;
					block_counter++;
				}
				else //-1
				{
					/*if(input_char == end_of_file)					//if EOF is detected then write current result to file
					{													
						if(write_counter != 0)
						{
							output_char = output_char << (8-write_counter);
							compressedfile_array[index_file]=output_char;
							output_char = 0x0;
						}
						else	//write_counter == 0
						{
							compressedfile_array[index_file]=output_char;
						}
					}*/

					break;
				}

				/*if(write_counter == 8)								//if result achieved 8 (size of char) then write it to compressed_file
				{
					compressedfile_array[index_file]=output_char;
					output_char = 0x0;
					write_counter = 0;
				}*/
			}
		
		block_cntr_array[index_blocks]=block_counter;
		//*d_last_byte_padding = write_counter;							//to decompress file we have to know how many bits in last byte have been written
		//update_config(write_counter); //TODO to zakomentowac przy ostatecznych pomiarach

		computearray_size(block_cntr_array,finalsize,orig_number_of_char);
		//final_compression(block_cntr_array,compressedfile_array,finalcompressed_array);

	}
	
}


void write_GPU_compressed(bool *final_compressed_cpu,int *finalsize_cpu)
{
FILE *compressed_file;
int write_counter=0;	//how many bits have been written in specific byte
//unsigned char input_char;
unsigned char output_char = 0x0;
//unsigned char end_of_file = 255;
unsigned char mask = 0x01; //00000001;

compressed_file = fopen(COMPRESSED_FILE_NAME_GPU, "wb");	

if ((compressed_file==NULL))
	{
		perror ("Error reading file");		
	}
else
	{
	for(int i = 0 ; i <  (*finalsize_cpu) ; i++)				
			{
			
			if(int(final_compressed_cpu[i]) == 0)
			{
				output_char = output_char << 1;					//if 0 then shift bits one position to left (last bit after shifting is 0)
					write_counter++;
			}
			else if (int(final_compressed_cpu[i]) == 1)
				{
					output_char = output_char << 1;					//if 1 then shift bits one position to left...
					output_char = output_char | mask;				//...and last bit change to: 1
					write_counter++;
				}
			if(write_counter == 8)								//if result achieved 8 (size of char) then write it to compressed_file
				{
					printf("Compressed char in decimal is %d \n", output_char);
					putc(output_char, compressed_file);				
					output_char = 0x0;
					write_counter = 0;
				}	
				
			}
		
		if(write_counter != 0)
						{
							output_char = output_char << (8-write_counter);
							printf("Compressed char in decimal is %d \n", output_char);
							putc(output_char, compressed_file);
							output_char = 0x0;
						}
	
	}
	fclose(compressed_file);
	last_byte_padding_gpu = write_counter;	
}

























void print_dchar_huffman_table()
{

		printf("\n dchar huffman table ");
		getchar();
		bool flag = false;

	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("Huffman table:\n");
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	
	for(int i = 0 ; i < MAX_CHAR ; i++)
	{		
		flag = false;
		for(int j = 0 ; j < (MAX_CHAR -1) ; j++)
		{		
			if(copiedarray2[i][j] != -1)
			{				
				if(!flag)
				{
					if(i == 10)//new line
					{
						printf("\\n:\t");
					}
					else
					{
						printf("%c:\t",i);
					}
				}
				flag = true;
				printf("%d ", copiedarray2[i][j]);
			}
		}
		if(flag) printf("\n");
	}

	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	getchar();
}

//Huffman table construction+++++++++++++++++++++++++++++++++++++++++++++++++++

void insertion_sort(node **forest, int length)
{	
	for(int i = 1; i < length ; i++)
	{	
		node *tmp = forest[i];
		int j = i - 1;
		bool done = false;

		do
		{
			if(forest[j]->weight < tmp->weight)		//> ascending order; < descending order
			{
				forest[j+1] = forest[j];
				j = j-1;
				if(j < 0)
				{
					done = true;
				}
			}
			else
			{
				done = true;
			}
		}while(!done);
		forest[j+1] = tmp;
	}
}

void print_char_frequency()
{
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("character frequency:\n");
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

	for(int i = 0 ; i < MAX_CHAR ; i++)
	{
		if(char_frequency[i] != 0)
		{
			if(i == 10)//new line
			{
				printf("%d)\tval: \\n\tfreq: %d\n",i, char_frequency[i]);
			}
			else
			{
				printf("%d)\tval: %c\tfreq: %d\n",i, i, char_frequency[i]);
			}
		}
	}

	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
}


void print_char_huffman_table()
{
	bool flag = false;

	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("Huffman table:\n");
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	
	for(int i = 0 ; i < MAX_CHAR ; i++)
	{		
		flag = false;
		for(int j = 0 ; j < (MAX_CHAR -1) ; j++)
		{		
			if(char_huffman_table[i][j] != -1)
			{				
				if(!flag)
				{
					if(i == 10)//new line
					{
						printf("\\n:\t");
					}
					else
					{
						printf("%c:\t",i);
					}
				}
				flag = true;
				printf("%d ", char_huffman_table[i][j]);
			}
		}
		if(flag) printf("\n");
	}

	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
}

void printout_inorder(node * tree) 
{	

	if(tree->left) printout_inorder(tree->left);

	if(tree->val != NULL)
	{
		if(tree->val == '\n')
		{
			printf("weight: %d\tvalue: \\n\n",tree->weight);
		}
		else
		{
			printf("weight: %d\tvalue: %c\n",tree->weight, tree->val);
		}
	}
	else
	{
		printf("weight: %d\tvalue: NULL\n",tree->weight);
	}

	if(tree->right) printout_inorder(tree->right);
}

void read_file()
{
	FILE *file;
	unsigned char end_of_file = 255;
	 unsigned char c;

	file = fopen(INPUT_FILE_NAME, "r");
  
	if (file==NULL)
	{
		perror ("Error reading file");		
	}
	else
	{
	//storing the file contents into h_input
		h_input = (unsigned char *)malloc(MAX_FILE_CHARS*sizeof(char));
		do
			{
			c = getc (file);
			//if(c == end_of_file) printf("\n Found EOF \n");
			//printf("c before putting into array is %c\n",c);
			h_input[number_of_char]=c;
			number_of_char++;
			char_frequency[c]++;			
		} while (c != end_of_file);

		fclose (file);		
	}
//	h_input[number_of_char] = end_of_file;
	char_frequency[end_of_file] = 0;				//to avoid problems with several EOF in one file
	//EOF is not needed ; so going to decrement
	number_of_char--;
}

void traverse_preorder(node *root, int *path)
{
	if(root->val != NULL)
	{
		for(int i = 0 ; i < MAX_CHAR -1 ; i++)
		{
			char_huffman_table[root->val][i] = path[i];
		}
	}

	if(root->left)//left 1
	{
		int counter = 0;

		for(int i = 0 ; i < MAX_CHAR - 1 ; i++)
		{			
			if(path[i] == -1)
			{
				break;
			}

			counter++;
		}

		path[counter] = 1;

		traverse_preorder(root->left, path);

		path[counter] = -1;
	}
	
	if(root->right)//right 0
	{
		int counter = 0;

		for(int i = 0 ; i < MAX_CHAR - 1 ; i++)
		{			
			if(path[i] == -1)
			{
				break;
			}

			counter++;
		}

		path[counter] = 0;

		traverse_preorder(root->right, path);

		path[counter] = -1;		
	}
}

void construct_huffman_table(node *root)
{		
	int path[MAX_CHAR - 1];
	for(int i = 0 ; i < MAX_CHAR - 1 ; i++)
	{
		path[i] = -1;
	}

	traverse_preorder(root, path);		
}


void build_binary_tree()
{
	int forest_counter = 0;
	node *forest[MAX_CHAR];
	node *curr;

	for(int i = 0 ; i < MAX_CHAR ; i++)		//initial forest
	{
		if(char_frequency[i] != 0)
		{
			curr = (node *)malloc(sizeof(node));
			curr->left = curr->right = NULL;
			curr->val = i;
			curr->weight = char_frequency[i];

			forest[forest_counter] = curr;
			forest_counter++;
		}
	}

	insertion_sort(forest, forest_counter);//sorted initial forest
		
	while(forest_counter > 1)//build final tree
	{
		node *parent;
		parent = (node *)malloc(sizeof(node));
		parent->right = forest[forest_counter-1];
		parent->left = forest[forest_counter-2];
		parent->weight = forest[forest_counter-1]->weight + forest[forest_counter-2]->weight;
		parent->val = NULL;

		forest[forest_counter-1] = NULL;
		forest[forest_counter-2] = parent;
	
		forest_counter--;

		insertion_sort(forest, forest_counter);
	}

	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("Huffman tree (inorder traversal sequence):\n");
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printout_inorder(forest[0]);
	printf("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");

	construct_huffman_table(forest[0]);

	//delete_binary_tree_postorder(forest[0]);	//after building Huffman table we do not need Huffman tree anymore
}

void array_initializer()
{
	for(int i = 0 ; i < MAX_CHAR ; i++)
	{
		char_frequency[i] = 0;
	}

	for(int i = 0 ; i < MAX_CHAR ; i++)
	{
		for(int j = 0 ; j < (MAX_CHAR-1) ; j++)
		{
			char_huffman_table[i][j] = -1;
		}
	}
}


//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//Calculation on CPU+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void compress_file()
{
	FILE *input_file;
	FILE *compressed_file;
	unsigned char output_char;
	int write_counter;	//how many bits have been written in specific byte
	
	unsigned char input_char;
	unsigned char end_of_file ;
	unsigned char mask ; //00000001;
	
	

	input_file = fopen(INPUT_FILE_NAME, "rb");
	compressed_file = fopen(COMPRESSED_FILE_NAME, "wb");			// apend file (add text to a file or create a file if it does not exist)
	output_char= 0x0;
	write_counter = 0;
	end_of_file = 255;
	mask = 0x01;

  
	if ((input_file==NULL)||(compressed_file==NULL))
	{
		perror ("Error reading file");		
	}
	else
	{
		do
		{
			input_char = getc (input_file);							//read one character from input file
			
			for(int i = 0 ; i < (MAX_CHAR - 1) ; i++)				
			{
				if(char_huffman_table[input_char][i] == 0)			//detect if current character on particular position has 0 or 1
				{
					output_char = output_char << 1;					//if 0 then shift bits one position to left (last bit after shifting is 0)
					write_counter++;
				}
				else if(char_huffman_table[input_char][i] == 1)
				{
					output_char = output_char << 1;					//if 1 then shift bits one position to left...
					output_char = output_char | mask;				//...and last bit change to: 1
					write_counter++;
				}
				else //-1
				{
					if(input_char == end_of_file)					//if EOF is detected then write current result to file
					{													
						if(write_counter != 0)
						{
							output_char = output_char << (8-write_counter);
							printf("Compressed char in decimal is %d \n", output_char);
							putc(output_char, compressed_file);
							output_char = 0x0;
						}
						else	//write_counter == 0
						{
							printf("Compressed char in decimal is %d \n", output_char);
							putc(output_char, compressed_file);				
						}
					}

					break;
				}

				if(write_counter == 8)								//if result achieved 8 (size of char) then write it to compressed_file
				{
					printf("Compressed char in decimal is %d \n", output_char);
					putc(output_char, compressed_file);				
					output_char = 0x0;
					write_counter = 0;
				}
			}

		} while (input_char != end_of_file);

		fclose (input_file);
		fclose(compressed_file);

		last_byte_padding = write_counter;							//to decompress file we have to know how many bits in last byte have been written
		
		//update_config(write_counter); //TODO to zakomentowac przy ostatecznych pomiarach
	}
}


void print_gpu_compressed_file(char *final_compressed_cpu,int finalsize_cpu)
{
FILE *compressed_file;

compressed_file = fopen(COMPRESSED_FILE_NAME_GPU, "wb");	
for(int i=0;i<finalsize_cpu;i++)
{

char c=final_compressed_cpu[i];
printf("i is %d and c is %c",i,c);
putc(c,compressed_file);

}

fclose(compressed_file);

}


void decompress_file()
{
	FILE *compressed_file;
	FILE *decompressed_file;
	unsigned char end_of_file = 255;
	unsigned char mask = 0x7F; //01111111;	
	unsigned char curr;
	unsigned char next;
	int written_char_counter=0;
	int pattern[MAX_CHAR - 1];

	for(int i = 0 ; i < (MAX_CHAR - 1); i++)
	{
		pattern[i] = -1;
	}

	compressed_file = fopen(COMPRESSED_FILE_NAME, "rb");
	decompressed_file = fopen(DECOMPRESSED_FILE_NAME, "wb");

	if ((compressed_file==NULL)||(decompressed_file==NULL))
	{
		perror ("Error reading file");		
	}
	else
	{
		int bit_counter=0;		
		unsigned char first_bit;
		bool read_next = true;

		curr = getc (compressed_file);
		next = getc (compressed_file);					//we have to read one byte in advance due to padding

//		for(int i = 0 ; i < (MAX_CHAR - 1) ; i++)		//builds a pattern and chcecks if it matches to char_huffman_table
		int pattern_counter=-1;
		while(pattern_counter < (MAX_CHAR - 1))
		{ 
			pattern_counter++;

			first_bit = curr | mask;							//check if first bit is 0 or 1

			curr = curr << 1;

			if(bit_counter == 7)
			{
				bit_counter = 0;				

				curr = next;

				if(read_next)
				{
					next = getc (compressed_file);

					if(next == end_of_file)
					{
						if((number_of_char - written_char_counter) < 8)
						{
							read_next = false;
							bit_counter = 7 - last_byte_padding;					
						}
					}
				}											

				if((curr == end_of_file) && ((number_of_char - written_char_counter) < 8))
				{
					break;
				}					
			}
			else
			{
				bit_counter++;
			}

			if(first_bit == 255)
			{
				pattern[pattern_counter] = 1;
			}
			else
			{
				pattern[pattern_counter] = 0;
			}

			bool flag = true;
			for(int j = 0 ; j < MAX_CHAR ; j++)
			{
				flag = true;

				for(int k = 0 ; k < (MAX_CHAR - 1) ; k++)
				{
					if(char_huffman_table[j][k] != pattern[k])
					{
						flag = false;
						break;
					}
				}

				if(flag == true)
				{	
					written_char_counter++;					
					putc(j, decompressed_file);
					
					for(int i = 0 ; i < (MAX_CHAR - 1); i++)
					{
						pattern[i] = -1;
					}

					pattern_counter = -1;

					break;
				}
			}
				
		}

		fclose (compressed_file);		
		fclose (decompressed_file);		
	}
}


void initialize()
{	
	array_initializer();	
	read_file();	
	print_char_frequency();
	build_binary_tree();
	print_char_huffman_table();
}


/*__global__ void compress(int *d_input,int number_of_char,int *d_char_huffman_table,int MAX_CHAR)
{
	int i=0;

	 extern __shared__ int my2DArray[32][32]; //size need to be coded a development time though  
	 my2DArray[threadIdx.x][threadIdx.y] = flatArray[blockDim.x * threadIdx.y + threadIdx.x];
	
}*/


__global__ void read2darray(int *devPtr,int pitch)
{
	int elements[2][2];
	 for (int r = 0; r < 2; ++r) {
        int* row = (int*)((char*)devPtr + r * pitch);
        for (int c = 0; c < 2; ++c) {
             elements[r][c] = row[c];
        }
    }
}


__global__ void check_bool(bool *d_bool)
{
	d_bool[0]=false;
	d_bool[1]=false;
}


void print_huffman()
{

printf(" \n Huffman after copying back \n " );
unsigned char input_char1;
unsigned char input_char2;

unsigned char input_char3;
unsigned char input_char4;

input_char1 = h_input[0];
input_char2 = h_input[1];
input_char3 = h_input[2];
input_char4 = h_input[3];
for (int i=0;i < MAX_CHAR-1;i++)
{
if(huffman_check[ input_char1*255+i]!= -1 )  printf ("\t%c code is  %d  \n", input_char1,huffman_check[ input_char1*255+i]);
}


for (int i=0;i < MAX_CHAR-1;i++)
{
if(huffman_check[ input_char2*255+i]!= -1) printf ("\t%c code is  %d  \n",input_char2, huffman_check[ input_char2*255+i]);
}

for (int i=0;i < MAX_CHAR-1;i++)
{
if(huffman_check[ input_char3*255+i]!= -1 )  printf ("\t%c code is  %d  \n", input_char3,huffman_check[ input_char3*255+i]);
}

for (int i=0;i < MAX_CHAR-1;i++)
{
if(huffman_check[ input_char4*255+i]!= -1 )  printf ("\t%c code is  %d  \n", input_char4,huffman_check[ input_char4*255+i]);
}




}

int main(int argc, char* argv[])
{
	
	int *finalsize_cpu=0;
	unsigned char end_of_file = 255;
	printf("start\n");
	initialize();

	cudaEvent_t start, stop;   	 	// cuda events to measure time
	float elapsed_time,elapsed_time_Cont; 

	cudaEventCreate(&start);     	// timing objects
	cudaEventCreate(&stop);

	unsigned int timer2=0;
	time_t seconds;


// In initialize in cpu,we put the file chars into array, fill huffman table and char_freq_arrays
//copy the input contents into an array
	cudaMalloc((void **)&d_input,number_of_char*sizeof(char));
	checkCUDAError("Error in allocating d_input");	
	cudaMemcpy(d_input,h_input,number_of_char*sizeof(char),cudaMemcpyHostToDevice);
	checkCUDAError("Error in copying d_input");	

	// Allocate space for the compressed file to be used in GPU
	cudaMalloc((void **)&compressedfile_array,number_of_char*(MAX_CHAR -1)*sizeof(bool));
	checkCUDAError("Error in allocating compressedfile_array");	
	//
	cudaMalloc((void **)&d_char_huffman_table,(MAX_CHAR)*(MAX_CHAR-1) * sizeof(int));
	checkCUDAError("Error in allocating d_char_huffman_table");	
	cudaMemcpy(d_char_huffman_table,char_huffman_table,(MAX_CHAR)*(MAX_CHAR-1) * sizeof(int),cudaMemcpyHostToDevice);
	checkCUDAError("Error in copying d_char_huffman_table");	
	cudaMemcpy(huffman_check,d_char_huffman_table,(MAX_CHAR)*(MAX_CHAR-1) * sizeof(int),cudaMemcpyDeviceToHost);
	checkCUDAError("Error in copying back");	

	cudaMalloc((void **)&block_cntr_array,number_of_char*sizeof(int));
	checkCUDAError("Error in allocating block_cntr_array");	

	cudaMalloc((void **)&d_last_byte_padding,sizeof(int));
	checkCUDAError("Error in allocating d_last_byte_padding");	
	cudaMalloc((void **)&finalsize,sizeof(int));
	checkCUDAError("Error in allocating finalsize");	
	cudaMalloc((void **)&orig_number_of_char,sizeof(int));
	checkCUDAError("Error in allocating orig_number_of_char");	
	cudaMemcpy(orig_number_of_char,&number_of_char,sizeof(int),cudaMemcpyHostToDevice);
	checkCUDAError("Error in copying orig_number_of_char");	

	
	
	// check if i can make a boolean array
	h_bool=(bool *) malloc(2*sizeof(bool));
	h_bool[0]=true;
	h_bool[1]=true;
	printf("bool1 is %d and bool2 is %d \n",h_bool[0],h_bool[1]);
	cudaMalloc((void **)&d_bool,2*sizeof(bool));
	checkCUDAError("Error in d_bool");	
	cudaMemcpy(d_bool,h_bool,2*sizeof(bool),cudaMemcpyHostToDevice);
	checkCUDAError("Error in copying d_bool");
	//check_bool<<<1,1>>>(d_bool);
	checkCUDAError("Error in kernel changing d_bool");
	cudaThreadSynchronize();
	checkCUDAError("Error in cudaThreadSynchronize");
	cudaMemcpy(h_bool,d_bool,2*sizeof(bool),cudaMemcpyDeviceToHost);
	checkCUDAError("Error in copying d_bool back");
	printf("Now bool1 is %d and bool2 is %d \n",h_bool[0],h_bool[1]);

	int checkhuff[2][3]= { {0, 0, 0},
							{1, 1, 1} };
	bool flag = true;
	for(int j=0;j<2;j++)
	{	
		flag = true;
			for(int k=0;k<2;k++)
			{
				printf("h_bool is %d \t checkhuff is %d \n",int(h_bool[k]),checkhuff[j][k]);
				if(checkhuff[j][k] != int(h_bool[k]))
				{
					flag = false;
					break;
				}
			}
			if(flag == true)
			{
				printf("pattern for %d is found\n",checkhuff[j][0]);
			}

	}
	


	//copy and send the huffman table as a 2d array to GPU Device
	//int *darray=0;
	//size_t pitch;   
	//cudaMallocPitch( (void**)&darray, &pitch, 2 * sizeof(int), 2); 
	//cudaMemcpy2D(darray,pitch,harray,2*sizeof(int),2*sizeof(int),2,cudaMemcpyHostToDevice);
	//cudaMalloc((void **)&darray,4*sizeof(int));
	//cudaMemcpy(darray,harray,4*sizeof(int),cudaMemcpyHostToDevice);
	//cudaMemcpy2D(copiedarray,2*sizeof(int),darray,pitch,pitch,2,cudaMemcpyDeviceToHost);
//	cudaMemcpy2D(copiedarray,2*sizeof(int),darray,pitch,2*sizeof(int),2,cudaMemcpyDeviceToHost);
	//printf("After copying back %d, \t %d, \t %d,  \t %d \n",copiedarray[0][0],copiedarray[0][1],copiedarray[1][0],copiedarray[1][1]);
	//int *darray_2d=0;
	//cudaMalloc((void **)&darray_2d,4*sizeof(int));
	//cudaMemcpy(darray,harray,4*sizeof(int),cudaMemcpyHostToDevice);
//	read2darray<<<1,1>>>(darray, pitch); 
	
	//size_t pitch2;   
	//cudaMallocPitch( (void**)&d_char_huffman_table, &pitch2, (MAX_CHAR-1) * sizeof(int), MAX_CHAR); 
	
	
	//cudaMemcpy2D(d_char_huffman_table,pitch2,char_huffman_table,(MAX_CHAR-1) * sizeof(int),(MAX_CHAR-1) * sizeof(int),MAX_CHAR,cudaMemcpyHostToDevice);
	
	//cudaMemcpy2D(char_huffman_table_gpu,(MAX_CHAR-1) * sizeof(int),char_huffman_table,(MAX_CHAR-1) * sizeof(int),(MAX_CHAR-1) * sizeof(int),MAX_CHAR,cudaMemcpyHostToDevice);
	//checkCUDAError("Error in char_huffman_table_gpu");	
	//cudaMemcpy2D(copiedarray2,(MAX_CHAR-1)*sizeof(int),d_char_huffman_table,pitch2,(MAX_CHAR-1)*sizeof(int),MAX_CHAR,cudaMemcpyDeviceToHost);
	
	//cudaMemcpy2D(copiedarray2,(MAX_CHAR-1)*sizeof(int),char_huffman_table_gpu,(MAX_CHAR-1) * sizeof(int),(MAX_CHAR-1) * sizeof(int),MAX_CHAR,cudaMemcpyDeviceToHost);
//cudaMemcpy(orig_number_of_char,&number_of_char,sizeof(int),cudaMemcpyHostToDevice);
//	checkCUDAError("Error in copiedarray2");	
//	print_dchar_huffman_table();
	printf("\n the number of characters in the input file is %d \n",number_of_char);
	getchar();
	/*for(int i=0;i<number_of_char;i++)
	{
		if( h_input[i] == end_of_file ) printf(" EOF \n");
		printf(" Copying into array: i is %d  and c is %c \n",i,h_input[i]);
	}*/
	


	getchar();
	print_huffman();
	
	int no_of_blocks = (number_of_char + BLOCK_SIZE -1)/BLOCK_SIZE;
		printf("no_of_blocksis %d \n", no_of_blocks);
	if(no_of_blocks == 0) no_of_blocks =1;

	//compress_file_gpu<<<number_of_char,1>>>(d_input,compressedfile_array,d_char_huffman_table,block_cntr_array,d_last_byte_padding);
	cudaEventRecord(start, 0);		// start time
	checkCUDAError("Error in cudaEventRecord start \n");	
	compress_file_gpu<<<no_of_blocks,BLOCK_SIZE>>>(d_input,compressedfile_array,d_char_huffman_table,block_cntr_array,d_last_byte_padding,finalsize,orig_number_of_char,number_of_char);
	checkCUDAError("Error in compress_file_gpu \n");	
	cudaThreadSynchronize();

	//cudaMalloc((void **)&block_cntr_array_check,number_of_char*sizeof(int));
	//checkCUDAError("Error in allocating block_cntr_array_check");	
	block_cntr_array_check = (int *) malloc(number_of_char*sizeof(int));
	
	cudaMemcpy(block_cntr_array_check,block_cntr_array,number_of_char*sizeof(int),cudaMemcpyDeviceToHost);
	checkCUDAError("Error in copying back block_cntr_array_check");

	for(int i=0; i < number_of_char; i++)
	{
		printf(" block size for i = %d is %d \n",i, block_cntr_array_check[i]);
	}


//	computearray_size<<<1,1>>>(block_cntr_array,finalsize,orig_number_of_char);
	checkCUDAError("Error in Compute array \n");
		finalsize_cpu = (int *)malloc(sizeof(int));
	cudaMemcpy(finalsize_cpu,finalsize,sizeof(int),cudaMemcpyDeviceToHost);
	printf("The final compressed array size is %d \n ", *finalsize_cpu);
	checkCUDAError("Error in finalsize_cpu");
	int block = *finalsize_cpu;
	//allocate space for the final compressed array

	cudaMalloc((void **)&finalcompressed_array,((*finalsize_cpu)*sizeof(bool)));
	checkCUDAError("cudaMemcpyHostToDevice");
	
	final_compression<<<no_of_blocks,BLOCK_SIZE>>>(block_cntr_array,compressedfile_array,finalcompressed_array,number_of_char);
	checkCUDAError("Error in final_compression call \n");

	cudaThreadSynchronize();
	checkCUDAError("Error in cudaThreadSynchronize \n");
	cudaEventRecord(stop, 0); 
	checkCUDAError("Error in cudaEventRecord stop \n");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("Time to calculate results: %f ms.\n", elapsed_time);  // print out execution time

	bool *final_compressed_cpu=0;
	final_compressed_cpu = (bool *)malloc((*finalsize_cpu)*sizeof(bool));
	cudaMemcpy(final_compressed_cpu,finalcompressed_array,((*finalsize_cpu)*sizeof(bool)),cudaMemcpyDeviceToHost);
	checkCUDAError("Error in copying final_compressed_cpu\n");
	//print_gpu_compressed_file(final_compressed_cpu,*finalsize_cpu);
	printf("The compressed value in binary is ");

	write_GPU_compressed(final_compressed_cpu,finalsize_cpu);
	for(int i=0;i<*finalsize_cpu;i++)
	//	sprintf(compress_file+i,final_compressed_cpu[i]);

	//printf("i is %d and val is %d \n",i,final_compressed_cpu[i]);
	printf("\n");




//	cudaMalloc((void **)&d_char_huffman_table,(MAX_CHAR)*(MAX_CHAR-1)*sizeof(int));
//	cudaMemcpy(d_char_huffman_table,char_huffman_table,(MAX_CHAR)*(MAX_CHAR-1)*sizeof(int),cudaMemcpyHostToDevice);
//	printf("\n Going to compress on the GPU ");
//	compress<<<number_of_char,1>>>(d_input,number_of_char,d_char_huffman_table,MAX_CHAR);

//test ends
	
	printf("compressing on CPU...\n");
	  //timer2=0;
	// CUT_SAFE_CALL(cutCreateTimer(&timer2));
	 //CUT_SAFE_CALL(cutStartTimer(timer2));

	/*clock_t Linuxclock_start,Linuxclock_end;		// clock return type
	cudaEvent_t CUDAevent_start, CUDAevent_end;
	float CUDAEvents_time; 
	cudaEventRecord(CUDAevent_start, 0 );
	cudaEventSynchronize(CUDAevent_start);  
	Linuxclock_start = clock();*/
	/*time_t before,after;

	before = time (NULL);*/
clock_t start1, stop1;
	start1 = clock();
	compress_file();
	stop1 = clock();
float elapsedTime = (float)(stop1 - start1) /
(float)CLOCKS_PER_SEC * 1000.0f;
printf( "Time in cpu : %3.1f ms\n", elapsedTime );
printf("Time to calculate results: %f ms.\n", elapsed_time);  // print out execution time

printf("Speedup achieved is %lf \n", elapsedTime/elapsed_time );
	/*after = time (NULL);
	double dif;
	dif = difftime (after,before);
	printf ("It took you %.9lf seconds to type your name.\n", dif );*/



	/*Linuxclock_end = clock();
	cudaEventRecord(CUDAevent_end, 0 );    	 // instrument code to measure end time
	cudaEventSynchronize(CUDAevent_end);
	cudaEventElapsedTime(&CUDAEvents_time, CUDAevent_start, CUDAevent_end);
	printf("CPU Time using CUDA events: %f ms\n", CUDAEvents_time);  // time_CUDAEvents is in ms
	printf("CPU Time using Linux clock: %f ms\n", ((double) (Linuxclock_end - Linuxclock_start) * 1000)/CLOCKS_PER_SEC);  //Linuxclock in sec
	cudaEventDestroy(CUDAevent_start);
	cudaEventDestroy(CUDAevent_end);*/

	//CUT_SAFE_CALL(cutStopTimer(timer2));
	//float time2=cutGetAverageTimerValue(timer2); 
	//printf("  Time on Host %f\n", time2);

	printf("decompressing on CPU...\n");
	decompress_file();

	getchar();
	return 0;
}
