#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
  float distance;
  int type;
} s_distance;

int sort_d(s_distance *arr, int elements);
float **parse_file(const char *filename, int *lines, int *features);
float euclidean_distance(float *u, float *v, int size);
void zero_array(int *arr, int size);

int main (int argc, char const* argv[]) {
  int training_lines, training_features;
  int testing_lines, testing_features;
  int k = atoi(argv[3]);

  float **training = parse_file(argv[1], &training_lines, &training_features);
  float **testing = parse_file(argv[2], &testing_lines, &testing_features);

  int i, j;

  s_distance **testing_distances = (s_distance **) malloc(sizeof(s_distance*) * testing_lines);

  #pragma omp parallel for shared(testing_distances) private(i, j)
  for (i = 0; i < testing_lines; i++) {
    testing_distances[i] = (s_distance *) malloc(sizeof(s_distance) * training_lines);
    for (j = 0; j < training_lines; j++) {
      testing_distances[i][j].distance = euclidean_distance(testing[i], training[j], training_features);
      testing_distances[i][j].type = training[j][training_features];
    }
  }

  #pragma omp barrier

  #pragma omp parallel for shared(testing_distances) private(i)
  for (i = 0; i < testing_lines; i++) {
    if (!sort_d(testing_distances[i], training_lines)) {
      perror("SORT FAILED");
      exit(1);
    }
  }

  #pragma omp barrier

  int *counts = (int *) malloc(sizeof(int) * training_lines);
  int errors = 0;

  for (i = 0; i < testing_lines; i++) {
    zero_array(counts, training_lines);

    int max = 0;
    for (j = 0; j < k; j++) {
      counts[testing_distances[i][j].type]++;
      if (counts[testing_distances[i][j].type] > counts[max])
        max = testing_distances[i][j].type;
    }

    int should = (int) testing[i][training_features];

    if (max != should)
      errors++;
  }

  printf("errors: %d (%.2f%%)\n", errors, ((float) errors / testing_lines) * 100);

  return 0;
}

void zero_array(int *arr, int size) {
  int i;

  #pragma omp parallel for private(i) shared(arr)
  for (i = 0; i < size; i++) {
    arr[i] = 0;
  }
}

float **parse_file(const char *filename, int *lines, int *features) {
  FILE *file = fopen(filename, "r");

  fscanf(file, "%d %d", lines, features);

  float **db = (float **) malloc(sizeof(float *) * *lines);

  int i, j;

  for (i = 0; i < *lines; i++) {
    db[i] = (float *) malloc(sizeof(float) * *features);

    for (j = 0; j <= *features; j++) {
      fscanf(file, "%f", db[i] + j);
    }
  }

  return db;
}

float euclidean_distance(float *u, float *v, int size) {
  int i;
  float sum = 0;
  float aux;

  for (i = 0; i < size; i++) {
    aux = v[i] - u[i];
    sum += aux * aux;
  }

  return sqrt(sum);
}

int sort_d(s_distance *arr, int elements) {
  #define  MAX_LEVELS  1000

  int beg[MAX_LEVELS], end[MAX_LEVELS], i = 0, L, R;
  s_distance piv;

  beg[0] = 0;
  end[0] = elements;

  while (i >= 0) {
    L = beg[i];
    R = end[i] - 1;

    if (L < R) {
      piv = arr[L];
      if (i == MAX_LEVELS-1)
          return 0;

      while (L < R) {
        while (arr[R].distance >= piv.distance && L < R)
          R--;
        if (L < R)
          arr[L++] = arr[R];

        while (arr[L].distance <= piv.distance && L < R)
          L++;
        if (L < R)
          arr[R--] = arr[L];
      }

      arr[L] = piv;
      beg[i + 1] = L + 1;
      end[i + 1] = end[i];
      end[i++] = L;
    }
    else {
      i--;
    }
  }
  return 1;
}

