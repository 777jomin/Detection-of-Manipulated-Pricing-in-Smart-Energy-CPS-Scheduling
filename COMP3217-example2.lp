/* Objective function */
max: xs;

xs=xs_11+xs_21+xs_22;
xs_11=x11_0+x11_1+x11_2+x11_3;
xs_21=x21_0+x21_1;
xs_22=x22_2+x22_3;
x11_0+x21_0=x0_t;
x11_1+x21_1=x1_t;
x11_2+x22_2=x2_t;
x11_3+x22_3=x3_t;
xt=x0_t+x1_t+x2_t+x3_t;
xs=xt;
xs_11<=2;
xs_21<=1;
xs_22<=1;
x11_0<=1;
x11_1<=1;
x11_2<=1;
x11_3<=1;
x21_0<=1;
x21_1<=1;
x22_2<=1;
x22_3<=1;
x0_t<=1;
x1_t<=1;
x2_t<=1;
x3_t<=1;
/* Variable bounds */