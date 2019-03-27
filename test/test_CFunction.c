#define bool _Bool
#define true 1
#define false 0
#define SIZE 100

double history_inputs[SIZE];
unsigned long n = 0;

void initfunc(void) {
  n = 0;
  for (int i=0; i<100; ++i) {
    history_inputs[i] = 0;
  }
  return;
}

void outputstep(const unsigned int portid, double *output, bool *valid) {
  if (portid == 0) {
    *output = history_inputs[(n-1)%SIZE];
    *valid = true;
    return;
  }
  if (portid == 1) {
    *valid = false;
    return;
  }
}

void blockstep(const double *x, double *y0, double *y1,
	       bool *valid0, bool* valid1) {
  *valid0 = false;
  *y1 = 2 * (*x);
  *valid1 = true;
  history_inputs[n%SIZE] = *x;
  ++n;
  return;
}

unsigned int n2 = 0;

void initfunc2(void) {
  n2 = 0;
  return;
}

void outputstep2(const unsigned int portid, double *output, bool *valid) {
  if (portid == 0) {
    *valid = false;
    return;
  }
  if (portid == 1) {
    *valid = false;
    return;
  }
}

void blockstep2(const double *x, double *y0, bool *valid0) {
  *valid0 = true;
  *y0 = x[0]+x[1];
  *(y0+1) = x[0]-x[1];
  ++n;
  return;
}
