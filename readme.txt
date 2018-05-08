18.05.07

SCCN with GUI

This network is different with Gong's,and it's a upgraded version of it.

1.The double-sides pre-train strategy of the last layer is different with others;
2.Using a soft decision function of Pu(map);
3.Update Pu(map) in time rather than wait until network parameters are fully.

18.05.08
1.Place the detection function and GUI in different threads to prevent the 
program from stuck when running.
