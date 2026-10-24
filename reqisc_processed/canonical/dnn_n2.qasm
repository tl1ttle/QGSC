OPENQASM 2.0;
include "qelib1.inc";

gate ryy(param0) q0,q1 {
    rx(pi/2) q0;
    rx(pi/2) q1;
    cx q0, q1;
    rz(param0) q1;
    cx q0, q1;
    rx(-pi/2) q0;
    rx(-pi/2) q1;
}
gate can (param0, param1, param2) q0,q1 {
    u3(1.5*pi, 0.0, 1.5*pi) q0;
    u3(0.5*pi, 1.5*pi, 0.5*pi) q1;
    cx q0, q1;
    u3(1.5*pi, param0 + pi, 0.5*pi) q0;
    u3(pi, 0.0, param1 + pi) q1;
    cx q0, q1;
    u3(0.5*pi, 0.0, 0.5*pi) q0;
    u3(0.0, 1.5*pi, param2 + 0.5*pi) q1;
    cx q0, q1;
}

// Qubits: [0, 1]
qreg q[2];

// Quantum gate operations
u3(0.5351290346716405*pi, 0.3442996898796907*pi, 0.019253321717914808*pi) q[1];
u3(0.3980395550762884*pi, 0.06501108094198897*pi, -0.6189481837132998*pi) q[0];
can(0.4764290594088865*pi, 0.22471062837232172*pi, -0.07223246909749956*pi) q[0], q[1];
u3(0.4371488914354579*pi, -0.948489233968651*pi, 0.8006160890552633*pi) q[1];
u3(0.7436219281399493*pi, 0.4142274235096023*pi, 0.8940551393307985*pi) q[0];
