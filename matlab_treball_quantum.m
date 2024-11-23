close all

min=10;
v=(-100:100);
z_vec=zeros(length(v), 1);
int_z_Y=0;

h_barra=1.054571596e-34;
c=physconst('LightSpeed');
G=6.67428e-11;
m_H=1.67262192369e-27;
m_sol=1.9885e+30;
M_P=sqrt(h_barra*c/G);
N_h=M_P/m_H;


figure(1)
hold on
for n=v

z_span=[10^-10 10];
inicial_cond=[1;n];
[z,Y]=ode45(@ODE, z_span, inicial_cond);

Y_1=Y(1:length(Y),1);

for k=1:length(Y_1)
    if abs(Y_1(k))<min
        min=abs(Y_1(k));
        index=k;
    end
end

if n==0
    z_der0=z;
    Y_1_der0=Y_1;
    index_der0=index;
end

z_vec(n+101)=z(index);
min=10;



plot(z,Y_1)


end
title('Value of Y(\zeta) for different initial values of its derivative')
hold off

z_0=sum(z_vec)/length(z_vec);

Y_pol_coef=polyfit(z_der0, Y_1_der0.^3.*z_der0.^2, 20); %% function to integrate
Y_pol=polyval(Y_pol_coef, z_der0);

Y_int_coef=polyint(Y_pol_coef);
Y_int=diff(polyval(Y_int_coef,[0 z_0]));

x=Y_1_der0(index_der0)*ones(length(z_der0),1); %% the straight line with "Y=0"

figure(2)
hold on
plot(z_der0,Y_1_der0,z_der0, x)
title("Numerical plot of Y")
xlabel("\zeta")
ylabel("Y")
hold off

figure(3)
hold on
plot(z_der0, Y_1_der0.^3.*z_der0.^2,z_der0, Y_pol, "-o")
title("Numerical and polynomial fit of the function Y^3\zeta^2")
xlabel("\zeta")
ylabel("Y")
legend("numerical", "polynomial")

M_c=sqrt(3*pi)/8*N_h^3*m_H*Y_int/m_sol;


function result = ODE(z, Y)
    %1/z^2*d/dz(z^2*dY/dz)+Y^3=0
    result=[Y(2); -Y(1).^3-2.*Y(2)./z];
end
