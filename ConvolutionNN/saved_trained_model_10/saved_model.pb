ящ	
г 
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.12v2.10.0-76-gfdfc646704c8ПЂ
ђ
Adam/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_24/bias/v
y
(Adam/dense_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/v*
_output_shapes
:
*
dtype0
і
Adam/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
║█
*'
shared_nameAdam/dense_24/kernel/v
Ѓ
*Adam/dense_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/v* 
_output_shapes
:
║█
*
dtype0
ѓ
Adam/conv2d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_35/bias/v
{
)Adam/conv2d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/v*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_35/kernel/v
І
+Adam/conv2d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/v*&
_output_shapes
:

*
dtype0
ѓ
Adam/conv2d_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_34/bias/v
{
)Adam/conv2d_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/v*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_34/kernel/v
І
+Adam/conv2d_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/v*&
_output_shapes
:

*
dtype0
ѓ
Adam/conv2d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_33/bias/v
{
)Adam/conv2d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/v*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_33/kernel/v
І
+Adam/conv2d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/v*&
_output_shapes
:

*
dtype0
ѓ
Adam/conv2d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_32/bias/v
{
)Adam/conv2d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/v*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_32/kernel/v
І
+Adam/conv2d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/v*&
_output_shapes
:
*
dtype0
ђ
Adam/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_24/bias/m
y
(Adam/dense_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/bias/m*
_output_shapes
:
*
dtype0
і
Adam/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
║█
*'
shared_nameAdam/dense_24/kernel/m
Ѓ
*Adam/dense_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_24/kernel/m* 
_output_shapes
:
║█
*
dtype0
ѓ
Adam/conv2d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_35/bias/m
{
)Adam/conv2d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/m*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_35/kernel/m
І
+Adam/conv2d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/m*&
_output_shapes
:

*
dtype0
ѓ
Adam/conv2d_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_34/bias/m
{
)Adam/conv2d_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/m*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_34/kernel/m
І
+Adam/conv2d_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/m*&
_output_shapes
:

*
dtype0
ѓ
Adam/conv2d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_33/bias/m
{
)Adam/conv2d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/m*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*(
shared_nameAdam/conv2d_33/kernel/m
І
+Adam/conv2d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/m*&
_output_shapes
:

*
dtype0
ѓ
Adam/conv2d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/conv2d_32/bias/m
{
)Adam/conv2d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/m*
_output_shapes
:
*
dtype0
њ
Adam/conv2d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/conv2d_32/kernel/m
І
+Adam/conv2d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/m*&
_output_shapes
:
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
:
*
dtype0
|
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
║█
* 
shared_namedense_24/kernel
u
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel* 
_output_shapes
:
║█
*
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:
*
dtype0
ё
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:
*
dtype0
ё
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:
*
dtype0
ё
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:

*
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
:
*
dtype0
ё
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
:
*
dtype0
ќ
serving_default_conv2d_32_inputPlaceholder*1
_output_shapes
:         ЯЯ*
dtype0*&
shape:         ЯЯ
Э
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_32_inputconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasdense_24/kerneldense_24/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference_signature_wrapper_77004

NoOpNoOp
┬V
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*§U
valueзUB­U BжU
├
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
╚
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op*
ј
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses* 
╚
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op*
ј
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
╚
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op*
╚
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op*
ј
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
ј
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
д
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias*
J
0
1
(2
)3
74
85
@6
A7
U8
V9*
J
0
1
(2
)3
74
85
@6
A7
U8
V9*
* 
░
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
\trace_0
]trace_1
^trace_2
_trace_3* 
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
* 
ї
diter

ebeta_1

fbeta_2
	gdecay
hlearning_ratem┤mх(mХ)mи7mИ8m╣@m║Am╗Um╝VmйvЙv┐(v└)v┴7v┬8v├@v─Av┼UvкVvК*

iserving_default* 

0
1*

0
1*
* 
Њ
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
`Z
VARIABLE_VALUEconv2d_32/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_32/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Љ
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

vtrace_0* 

wtrace_0* 

(0
)1*

(0
)1*
* 
Њ
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

}trace_0* 

~trace_0* 
`Z
VARIABLE_VALUEconv2d_33/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_33/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ћ
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

ёtrace_0* 

Ёtrace_0* 

70
81*

70
81*
* 
ў
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Іtrace_0* 

їtrace_0* 
`Z
VARIABLE_VALUEconv2d_34/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_34/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

@0
A1*

@0
A1*
* 
ў
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

њtrace_0* 

Њtrace_0* 
`Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
ќ
ћnon_trainable_variables
Ћlayers
ќmetrics
 Ќlayer_regularization_losses
ўlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

Ўtrace_0* 

џtrace_0* 
* 
* 
* 
ќ
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

аtrace_0* 

Аtrace_0* 

U0
V1*

U0
V1*
* 
ў
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

Дtrace_0* 

еtrace_0* 
_Y
VARIABLE_VALUEdense_24/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_24/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*

Е0
ф1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ф	variables
г	keras_api

Гtotal

«count*
M
»	variables
░	keras_api

▒total

▓count
│
_fn_kwargs*

Г0
«1*

Ф	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

▒0
▓1*

»	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
Ѓ}
VARIABLE_VALUEAdam/conv2d_32/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_32/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_33/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_33/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_34/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_34/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_35/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_35/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_24/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_32/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_32/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_33/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_33/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_34/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_34/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѓ}
VARIABLE_VALUEAdam/conv2d_35/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_35/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_24/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_24/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╣
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_32/kernel/m/Read/ReadVariableOp)Adam/conv2d_32/bias/m/Read/ReadVariableOp+Adam/conv2d_33/kernel/m/Read/ReadVariableOp)Adam/conv2d_33/bias/m/Read/ReadVariableOp+Adam/conv2d_34/kernel/m/Read/ReadVariableOp)Adam/conv2d_34/bias/m/Read/ReadVariableOp+Adam/conv2d_35/kernel/m/Read/ReadVariableOp)Adam/conv2d_35/bias/m/Read/ReadVariableOp*Adam/dense_24/kernel/m/Read/ReadVariableOp(Adam/dense_24/bias/m/Read/ReadVariableOp+Adam/conv2d_32/kernel/v/Read/ReadVariableOp)Adam/conv2d_32/bias/v/Read/ReadVariableOp+Adam/conv2d_33/kernel/v/Read/ReadVariableOp)Adam/conv2d_33/bias/v/Read/ReadVariableOp+Adam/conv2d_34/kernel/v/Read/ReadVariableOp)Adam/conv2d_34/bias/v/Read/ReadVariableOp+Adam/conv2d_35/kernel/v/Read/ReadVariableOp)Adam/conv2d_35/bias/v/Read/ReadVariableOp*Adam/dense_24/kernel/v/Read/ReadVariableOp(Adam/dense_24/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *'
f"R 
__inference__traced_save_77420
е
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasdense_24/kerneldense_24/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_32/kernel/mAdam/conv2d_32/bias/mAdam/conv2d_33/kernel/mAdam/conv2d_33/bias/mAdam/conv2d_34/kernel/mAdam/conv2d_34/bias/mAdam/conv2d_35/kernel/mAdam/conv2d_35/bias/mAdam/dense_24/kernel/mAdam/dense_24/bias/mAdam/conv2d_32/kernel/vAdam/conv2d_32/bias/vAdam/conv2d_33/kernel/vAdam/conv2d_33/bias/vAdam/conv2d_34/kernel/vAdam/conv2d_34/bias/vAdam/conv2d_35/kernel/vAdam/conv2d_35/bias/vAdam/dense_24/kernel/vAdam/dense_24/bias/v*3
Tin,
*2(*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_restore_77547╬┬
▓

§
D__inference_conv2d_32_layer_call_and_return_conditional_losses_77159

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         яя
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_77249

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ш
ъ
)__inference_conv2d_32_layer_call_fn_77149

inputs!
unknown:

	unknown_0:

identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_76620y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         яя
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
М

ћ
-__inference_sequential_14_layer_call_fn_77054

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:


	unknown_4:
#
	unknown_5:


	unknown_6:

	unknown_7:
║█

	unknown_8:

identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_76857o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
╔
a
E__inference_flatten_14_layer_call_and_return_conditional_losses_76692

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ║m  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ║█Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ║█"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         55
:W S
/
_output_shapes
:         55

 
_user_specified_nameinputs
Е(
Ї
H__inference_sequential_14_layer_call_and_return_conditional_losses_76971
conv2d_32_input)
conv2d_32_76941:

conv2d_32_76943:
)
conv2d_33_76947:


conv2d_33_76949:
)
conv2d_34_76953:


conv2d_34_76955:
)
conv2d_35_76958:


conv2d_35_76960:
"
dense_24_76965:
║█

dense_24_76967:

identityѕб!conv2d_32/StatefulPartitionedCallб!conv2d_33/StatefulPartitionedCallб!conv2d_34/StatefulPartitionedCallб!conv2d_35/StatefulPartitionedCallб dense_24/StatefulPartitionedCallЄ
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputconv2d_32_76941conv2d_32_76943*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_76620­
activation_1/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_76631Ю
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_33_76947conv2d_33_76949*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ▄▄
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_76644Ш
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_76588Ъ
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_34_76953conv2d_34_76955*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_76662а
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_76958conv2d_35_76960*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_76679Ш
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_76600с
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ║█* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_14_layer_call_and_return_conditional_losses_76692Ї
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0dense_24_76965dense_24_76967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_76705x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
щ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:b ^
1
_output_shapes
:         ЯЯ
)
_user_specified_nameconv2d_32_input
╗
L
0__inference_max_pooling2d_20_layer_call_fn_77194

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_76588Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь

Ю
-__inference_sequential_14_layer_call_fn_76735
conv2d_32_input!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:


	unknown_4:
#
	unknown_5:


	unknown_6:

	unknown_7:
║█

	unknown_8:

identityѕбStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_76712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         ЯЯ
)
_user_specified_nameconv2d_32_input
љ
§
D__inference_conv2d_33_layer_call_and_return_conditional_losses_77189

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ▄▄
k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ▄▄
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         яя
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         яя

 
_user_specified_nameinputs
ё5
Д
H__inference_sequential_14_layer_call_and_return_conditional_losses_77140

inputsB
(conv2d_32_conv2d_readvariableop_resource:
7
)conv2d_32_biasadd_readvariableop_resource:
B
(conv2d_33_conv2d_readvariableop_resource:

7
)conv2d_33_biasadd_readvariableop_resource:
B
(conv2d_34_conv2d_readvariableop_resource:

7
)conv2d_34_biasadd_readvariableop_resource:
B
(conv2d_35_conv2d_readvariableop_resource:

7
)conv2d_35_biasadd_readvariableop_resource:
;
'dense_24_matmul_readvariableop_resource:
║█
6
(dense_24_biasadd_readvariableop_resource:

identityѕб conv2d_32/BiasAdd/ReadVariableOpбconv2d_32/Conv2D/ReadVariableOpб conv2d_33/BiasAdd/ReadVariableOpбconv2d_33/Conv2D/ReadVariableOpб conv2d_34/BiasAdd/ReadVariableOpбconv2d_34/Conv2D/ReadVariableOpб conv2d_35/BiasAdd/ReadVariableOpбconv2d_35/Conv2D/ReadVariableOpбdense_24/BiasAdd/ReadVariableOpбdense_24/MatMul/ReadVariableOpљ
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0░
conv2d_32/Conv2DConv2Dinputs'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
*
paddingVALID*
strides
є
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ю
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
q
activation_1/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:         яя
љ
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╔
conv2d_33/Conv2DConv2Dactivation_1/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
*
paddingVALID*
strides
є
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ю
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
n
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:         ▄▄
«
max_pooling2d_20/MaxPoolMaxPoolconv2d_33/Relu:activations:0*/
_output_shapes
:         nn
*
ksize
*
paddingVALID*
strides
љ
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╔
conv2d_34/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
*
paddingVALID*
strides
є
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Џ
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
l
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:         ll
љ
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0─
conv2d_35/Conv2DConv2Dconv2d_34/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
*
paddingVALID*
strides
є
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Џ
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
l
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         jj
«
max_pooling2d_21/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:         55
*
ksize
*
paddingVALID*
strides
a
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ║m  Ј
flatten_14/ReshapeReshape!max_pooling2d_21/MaxPool:output:0flatten_14/Const:output:0*
T0*)
_output_shapes
:         ║█ѕ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
║█
*
dtype0љ
dense_24/MatMulMatMulflatten_14/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
ё
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Љ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_24/SoftmaxSoftmaxdense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_24/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Ю
NoOpNoOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
М

ћ
-__inference_sequential_14_layer_call_fn_77029

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:


	unknown_4:
#
	unknown_5:


	unknown_6:

	unknown_7:
║█

	unknown_8:

identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_76712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
ё
§
D__inference_conv2d_34_layer_call_and_return_conditional_losses_76662

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ll
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ll
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         nn
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         nn

 
_user_specified_nameinputs
╗
L
0__inference_max_pooling2d_21_layer_call_fn_77244

inputs
identity▄
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_76600Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_76600

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь

Ю
-__inference_sequential_14_layer_call_fn_76905
conv2d_32_input!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:


	unknown_4:
#
	unknown_5:


	unknown_6:

	unknown_7:
║█

	unknown_8:

identityѕбStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_sequential_14_layer_call_and_return_conditional_losses_76857o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         ЯЯ
)
_user_specified_nameconv2d_32_input
╝

Њ
#__inference_signature_wrapper_77004
conv2d_32_input!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:


	unknown_4:
#
	unknown_5:


	unknown_6:

	unknown_7:
║█

	unknown_8:

identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__wrapped_model_76579o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
1
_output_shapes
:         ЯЯ
)
_user_specified_nameconv2d_32_input
ј(
ё
H__inference_sequential_14_layer_call_and_return_conditional_losses_76857

inputs)
conv2d_32_76827:

conv2d_32_76829:
)
conv2d_33_76833:


conv2d_33_76835:
)
conv2d_34_76839:


conv2d_34_76841:
)
conv2d_35_76844:


conv2d_35_76846:
"
dense_24_76851:
║█

dense_24_76853:

identityѕб!conv2d_32/StatefulPartitionedCallб!conv2d_33/StatefulPartitionedCallб!conv2d_34/StatefulPartitionedCallб!conv2d_35/StatefulPartitionedCallб dense_24/StatefulPartitionedCall■
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32_76827conv2d_32_76829*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_76620­
activation_1/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_76631Ю
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_33_76833conv2d_33_76835*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ▄▄
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_76644Ш
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_76588Ъ
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_34_76839conv2d_34_76841*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_76662а
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_76844conv2d_35_76846*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_76679Ш
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_76600с
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ║█* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_14_layer_call_and_return_conditional_losses_76692Ї
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0dense_24_76851dense_24_76853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_76705x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
щ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
ь
ъ
)__inference_conv2d_35_layer_call_fn_77228

inputs!
unknown:


	unknown_0:

identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_76679w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         jj
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ll
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         ll

 
_user_specified_nameinputs
ё5
Д
H__inference_sequential_14_layer_call_and_return_conditional_losses_77097

inputsB
(conv2d_32_conv2d_readvariableop_resource:
7
)conv2d_32_biasadd_readvariableop_resource:
B
(conv2d_33_conv2d_readvariableop_resource:

7
)conv2d_33_biasadd_readvariableop_resource:
B
(conv2d_34_conv2d_readvariableop_resource:

7
)conv2d_34_biasadd_readvariableop_resource:
B
(conv2d_35_conv2d_readvariableop_resource:

7
)conv2d_35_biasadd_readvariableop_resource:
;
'dense_24_matmul_readvariableop_resource:
║█
6
(dense_24_biasadd_readvariableop_resource:

identityѕб conv2d_32/BiasAdd/ReadVariableOpбconv2d_32/Conv2D/ReadVariableOpб conv2d_33/BiasAdd/ReadVariableOpбconv2d_33/Conv2D/ReadVariableOpб conv2d_34/BiasAdd/ReadVariableOpбconv2d_34/Conv2D/ReadVariableOpб conv2d_35/BiasAdd/ReadVariableOpбconv2d_35/Conv2D/ReadVariableOpбdense_24/BiasAdd/ReadVariableOpбdense_24/MatMul/ReadVariableOpљ
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0░
conv2d_32/Conv2DConv2Dinputs'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
*
paddingVALID*
strides
є
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ю
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
q
activation_1/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:         яя
љ
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╔
conv2d_33/Conv2DConv2Dactivation_1/Relu:activations:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
*
paddingVALID*
strides
є
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Ю
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
n
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:         ▄▄
«
max_pooling2d_20/MaxPoolMaxPoolconv2d_33/Relu:activations:0*/
_output_shapes
:         nn
*
ksize
*
paddingVALID*
strides
љ
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0╔
conv2d_34/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
*
paddingVALID*
strides
є
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Џ
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
l
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:         ll
љ
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0─
conv2d_35/Conv2DConv2Dconv2d_34/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
*
paddingVALID*
strides
є
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Џ
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
l
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         jj
«
max_pooling2d_21/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:         55
*
ksize
*
paddingVALID*
strides
a
flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ║m  Ј
flatten_14/ReshapeReshape!max_pooling2d_21/MaxPool:output:0flatten_14/Const:output:0*
T0*)
_output_shapes
:         ║█ѕ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource* 
_output_shapes
:
║█
*
dtype0љ
dense_24/MatMulMatMulflatten_14/Reshape:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
ё
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Љ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
h
dense_24/SoftmaxSoftmaxdense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         
i
IdentityIdentitydense_24/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Ю
NoOpNoOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^dense_24/BiasAdd/ReadVariableOp^dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
dense_24/BiasAdd/ReadVariableOpdense_24/BiasAdd/ReadVariableOp2@
dense_24/MatMul/ReadVariableOpdense_24/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
Њ
g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_77199

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
­A
а

 __inference__wrapped_model_76579
conv2d_32_inputP
6sequential_14_conv2d_32_conv2d_readvariableop_resource:
E
7sequential_14_conv2d_32_biasadd_readvariableop_resource:
P
6sequential_14_conv2d_33_conv2d_readvariableop_resource:

E
7sequential_14_conv2d_33_biasadd_readvariableop_resource:
P
6sequential_14_conv2d_34_conv2d_readvariableop_resource:

E
7sequential_14_conv2d_34_biasadd_readvariableop_resource:
P
6sequential_14_conv2d_35_conv2d_readvariableop_resource:

E
7sequential_14_conv2d_35_biasadd_readvariableop_resource:
I
5sequential_14_dense_24_matmul_readvariableop_resource:
║█
D
6sequential_14_dense_24_biasadd_readvariableop_resource:

identityѕб.sequential_14/conv2d_32/BiasAdd/ReadVariableOpб-sequential_14/conv2d_32/Conv2D/ReadVariableOpб.sequential_14/conv2d_33/BiasAdd/ReadVariableOpб-sequential_14/conv2d_33/Conv2D/ReadVariableOpб.sequential_14/conv2d_34/BiasAdd/ReadVariableOpб-sequential_14/conv2d_34/Conv2D/ReadVariableOpб.sequential_14/conv2d_35/BiasAdd/ReadVariableOpб-sequential_14/conv2d_35/Conv2D/ReadVariableOpб-sequential_14/dense_24/BiasAdd/ReadVariableOpб,sequential_14/dense_24/MatMul/ReadVariableOpг
-sequential_14/conv2d_32/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0Н
sequential_14/conv2d_32/Conv2DConv2Dconv2d_32_input5sequential_14/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
*
paddingVALID*
strides
б
.sequential_14/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0К
sequential_14/conv2d_32/BiasAddBiasAdd'sequential_14/conv2d_32/Conv2D:output:06sequential_14/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
Ї
sequential_14/activation_1/ReluRelu(sequential_14/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:         яя
г
-sequential_14/conv2d_33/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0з
sequential_14/conv2d_33/Conv2DConv2D-sequential_14/activation_1/Relu:activations:05sequential_14/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
*
paddingVALID*
strides
б
.sequential_14/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0К
sequential_14/conv2d_33/BiasAddBiasAdd'sequential_14/conv2d_33/Conv2D:output:06sequential_14/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
і
sequential_14/conv2d_33/ReluRelu(sequential_14/conv2d_33/BiasAdd:output:0*
T0*1
_output_shapes
:         ▄▄
╩
&sequential_14/max_pooling2d_20/MaxPoolMaxPool*sequential_14/conv2d_33/Relu:activations:0*/
_output_shapes
:         nn
*
ksize
*
paddingVALID*
strides
г
-sequential_14/conv2d_34/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0з
sequential_14/conv2d_34/Conv2DConv2D/sequential_14/max_pooling2d_20/MaxPool:output:05sequential_14/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
*
paddingVALID*
strides
б
.sequential_14/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0┼
sequential_14/conv2d_34/BiasAddBiasAdd'sequential_14/conv2d_34/Conv2D:output:06sequential_14/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
ѕ
sequential_14/conv2d_34/ReluRelu(sequential_14/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:         ll
г
-sequential_14/conv2d_35/Conv2D/ReadVariableOpReadVariableOp6sequential_14_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0Ь
sequential_14/conv2d_35/Conv2DConv2D*sequential_14/conv2d_34/Relu:activations:05sequential_14/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
*
paddingVALID*
strides
б
.sequential_14/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp7sequential_14_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0┼
sequential_14/conv2d_35/BiasAddBiasAdd'sequential_14/conv2d_35/Conv2D:output:06sequential_14/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
ѕ
sequential_14/conv2d_35/ReluRelu(sequential_14/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:         jj
╩
&sequential_14/max_pooling2d_21/MaxPoolMaxPool*sequential_14/conv2d_35/Relu:activations:0*/
_output_shapes
:         55
*
ksize
*
paddingVALID*
strides
o
sequential_14/flatten_14/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ║m  ╣
 sequential_14/flatten_14/ReshapeReshape/sequential_14/max_pooling2d_21/MaxPool:output:0'sequential_14/flatten_14/Const:output:0*
T0*)
_output_shapes
:         ║█ц
,sequential_14/dense_24/MatMul/ReadVariableOpReadVariableOp5sequential_14_dense_24_matmul_readvariableop_resource* 
_output_shapes
:
║█
*
dtype0║
sequential_14/dense_24/MatMulMatMul)sequential_14/flatten_14/Reshape:output:04sequential_14/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
а
-sequential_14/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_14_dense_24_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0╗
sequential_14/dense_24/BiasAddBiasAdd'sequential_14/dense_24/MatMul:product:05sequential_14/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
ё
sequential_14/dense_24/SoftmaxSoftmax'sequential_14/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         
w
IdentityIdentity(sequential_14/dense_24/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
Е
NoOpNoOp/^sequential_14/conv2d_32/BiasAdd/ReadVariableOp.^sequential_14/conv2d_32/Conv2D/ReadVariableOp/^sequential_14/conv2d_33/BiasAdd/ReadVariableOp.^sequential_14/conv2d_33/Conv2D/ReadVariableOp/^sequential_14/conv2d_34/BiasAdd/ReadVariableOp.^sequential_14/conv2d_34/Conv2D/ReadVariableOp/^sequential_14/conv2d_35/BiasAdd/ReadVariableOp.^sequential_14/conv2d_35/Conv2D/ReadVariableOp.^sequential_14/dense_24/BiasAdd/ReadVariableOp-^sequential_14/dense_24/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2`
.sequential_14/conv2d_32/BiasAdd/ReadVariableOp.sequential_14/conv2d_32/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_32/Conv2D/ReadVariableOp-sequential_14/conv2d_32/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_33/BiasAdd/ReadVariableOp.sequential_14/conv2d_33/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_33/Conv2D/ReadVariableOp-sequential_14/conv2d_33/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_34/BiasAdd/ReadVariableOp.sequential_14/conv2d_34/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_34/Conv2D/ReadVariableOp-sequential_14/conv2d_34/Conv2D/ReadVariableOp2`
.sequential_14/conv2d_35/BiasAdd/ReadVariableOp.sequential_14/conv2d_35/BiasAdd/ReadVariableOp2^
-sequential_14/conv2d_35/Conv2D/ReadVariableOp-sequential_14/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_14/dense_24/BiasAdd/ReadVariableOp-sequential_14/dense_24/BiasAdd/ReadVariableOp2\
,sequential_14/dense_24/MatMul/ReadVariableOp,sequential_14/dense_24/MatMul/ReadVariableOp:b ^
1
_output_shapes
:         ЯЯ
)
_user_specified_nameconv2d_32_input
з
c
G__inference_activation_1_layer_call_and_return_conditional_losses_77169

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:         яя
d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         яя
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         яя
:Y U
1
_output_shapes
:         яя

 
_user_specified_nameinputs
╬
H
,__inference_activation_1_layer_call_fn_77164

inputs
identity┐
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_76631j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         яя
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         яя
:Y U
1
_output_shapes
:         яя

 
_user_specified_nameinputs
ј(
ё
H__inference_sequential_14_layer_call_and_return_conditional_losses_76712

inputs)
conv2d_32_76621:

conv2d_32_76623:
)
conv2d_33_76645:


conv2d_33_76647:
)
conv2d_34_76663:


conv2d_34_76665:
)
conv2d_35_76680:


conv2d_35_76682:
"
dense_24_76706:
║█

dense_24_76708:

identityѕб!conv2d_32/StatefulPartitionedCallб!conv2d_33/StatefulPartitionedCallб!conv2d_34/StatefulPartitionedCallб!conv2d_35/StatefulPartitionedCallб dense_24/StatefulPartitionedCall■
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32_76621conv2d_32_76623*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_76620­
activation_1/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_76631Ю
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_33_76645conv2d_33_76647*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ▄▄
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_76644Ш
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_76588Ъ
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_34_76663conv2d_34_76665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_76662а
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_76680conv2d_35_76682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_76679Ш
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_76600с
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ║█* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_14_layer_call_and_return_conditional_losses_76692Ї
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0dense_24_76706dense_24_76708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_76705x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
щ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
▄ю
Ъ
!__inference__traced_restore_77547
file_prefix;
!assignvariableop_conv2d_32_kernel:
/
!assignvariableop_1_conv2d_32_bias:
=
#assignvariableop_2_conv2d_33_kernel:

/
!assignvariableop_3_conv2d_33_bias:
=
#assignvariableop_4_conv2d_34_kernel:

/
!assignvariableop_5_conv2d_34_bias:
=
#assignvariableop_6_conv2d_35_kernel:

/
!assignvariableop_7_conv2d_35_bias:
6
"assignvariableop_8_dense_24_kernel:
║█
.
 assignvariableop_9_dense_24_bias:
'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: E
+assignvariableop_19_adam_conv2d_32_kernel_m:
7
)assignvariableop_20_adam_conv2d_32_bias_m:
E
+assignvariableop_21_adam_conv2d_33_kernel_m:

7
)assignvariableop_22_adam_conv2d_33_bias_m:
E
+assignvariableop_23_adam_conv2d_34_kernel_m:

7
)assignvariableop_24_adam_conv2d_34_bias_m:
E
+assignvariableop_25_adam_conv2d_35_kernel_m:

7
)assignvariableop_26_adam_conv2d_35_bias_m:
>
*assignvariableop_27_adam_dense_24_kernel_m:
║█
6
(assignvariableop_28_adam_dense_24_bias_m:
E
+assignvariableop_29_adam_conv2d_32_kernel_v:
7
)assignvariableop_30_adam_conv2d_32_bias_v:
E
+assignvariableop_31_adam_conv2d_33_kernel_v:

7
)assignvariableop_32_adam_conv2d_33_bias_v:
E
+assignvariableop_33_adam_conv2d_34_kernel_v:

7
)assignvariableop_34_adam_conv2d_34_bias_v:
E
+assignvariableop_35_adam_conv2d_35_kernel_v:

7
)assignvariableop_36_adam_conv2d_35_bias_v:
>
*assignvariableop_37_adam_dense_24_kernel_v:
║█
6
(assignvariableop_38_adam_dense_24_bias_v:

identity_40ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9В
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*њ
valueѕBЁ(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH└
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ж
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_33_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_34_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_34_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_35_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_35_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_24_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_24_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:ј
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_32_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_32_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_33_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_33_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_34_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_34_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_35_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_35_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_24_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_24_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_32_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_32_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_33_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_33_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_34_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_34_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_35_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_35_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_dense_24_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_dense_24_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Е
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_40IdentityIdentity_39:output:0^NoOp_1*
T0*
_output_shapes
: ќ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_40Identity_40:output:0*c
_input_shapesR
P: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
▓

§
D__inference_conv2d_32_layer_call_and_return_conditional_losses_76620

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         яя
i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         яя
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЯЯ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЯЯ
 
_user_specified_nameinputs
љ
§
D__inference_conv2d_33_layer_call_and_return_conditional_losses_76644

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ▄▄
Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ▄▄
k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ▄▄
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         яя
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         яя

 
_user_specified_nameinputs
ё
§
D__inference_conv2d_35_layer_call_and_return_conditional_losses_77239

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         jj
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         jj
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ll
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ll

 
_user_specified_nameinputs
ё
§
D__inference_conv2d_35_layer_call_and_return_conditional_losses_76679

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         jj
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         jj
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         jj
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ll
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ll

 
_user_specified_nameinputs
Х
F
*__inference_flatten_14_layer_call_fn_77254

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ║█* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_14_layer_call_and_return_conditional_losses_76692b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ║█"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         55
:W S
/
_output_shapes
:         55

 
_user_specified_nameinputs
ЦR
Ї
__inference__traced_save_77420
file_prefix/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_32_kernel_m_read_readvariableop4
0savev2_adam_conv2d_32_bias_m_read_readvariableop6
2savev2_adam_conv2d_33_kernel_m_read_readvariableop4
0savev2_adam_conv2d_33_bias_m_read_readvariableop6
2savev2_adam_conv2d_34_kernel_m_read_readvariableop4
0savev2_adam_conv2d_34_bias_m_read_readvariableop6
2savev2_adam_conv2d_35_kernel_m_read_readvariableop4
0savev2_adam_conv2d_35_bias_m_read_readvariableop5
1savev2_adam_dense_24_kernel_m_read_readvariableop3
/savev2_adam_dense_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_32_kernel_v_read_readvariableop4
0savev2_adam_conv2d_32_bias_v_read_readvariableop6
2savev2_adam_conv2d_33_kernel_v_read_readvariableop4
0savev2_adam_conv2d_33_bias_v_read_readvariableop6
2savev2_adam_conv2d_34_kernel_v_read_readvariableop4
0savev2_adam_conv2d_34_bias_v_read_readvariableop6
2savev2_adam_conv2d_35_kernel_v_read_readvariableop4
0savev2_adam_conv2d_35_bias_v_read_readvariableop5
1savev2_adam_dense_24_kernel_v_read_readvariableop3
/savev2_adam_dense_24_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ж
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*њ
valueѕBЁ(B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHй
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B П
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_32_kernel_m_read_readvariableop0savev2_adam_conv2d_32_bias_m_read_readvariableop2savev2_adam_conv2d_33_kernel_m_read_readvariableop0savev2_adam_conv2d_33_bias_m_read_readvariableop2savev2_adam_conv2d_34_kernel_m_read_readvariableop0savev2_adam_conv2d_34_bias_m_read_readvariableop2savev2_adam_conv2d_35_kernel_m_read_readvariableop0savev2_adam_conv2d_35_bias_m_read_readvariableop1savev2_adam_dense_24_kernel_m_read_readvariableop/savev2_adam_dense_24_bias_m_read_readvariableop2savev2_adam_conv2d_32_kernel_v_read_readvariableop0savev2_adam_conv2d_32_bias_v_read_readvariableop2savev2_adam_conv2d_33_kernel_v_read_readvariableop0savev2_adam_conv2d_33_bias_v_read_readvariableop2savev2_adam_conv2d_34_kernel_v_read_readvariableop0savev2_adam_conv2d_34_bias_v_read_readvariableop2savev2_adam_conv2d_35_kernel_v_read_readvariableop0savev2_adam_conv2d_35_bias_v_read_readvariableop1savev2_adam_dense_24_kernel_v_read_readvariableop/savev2_adam_dense_24_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ђ
_input_shapes№
В: :
:
:

:
:

:
:

:
:
║█
:
: : : : : : : : : :
:
:

:
:

:
:

:
:
║█
:
:
:
:

:
:

:
:

:
:
║█
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:&	"
 
_output_shapes
:
║█
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:&"
 
_output_shapes
:
║█
: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
:
:, (
&
_output_shapes
:

: !

_output_shapes
:
:,"(
&
_output_shapes
:

: #

_output_shapes
:
:,$(
&
_output_shapes
:

: %

_output_shapes
:
:&&"
 
_output_shapes
:
║█
: '

_output_shapes
:
:(

_output_shapes
: 
Њ
g
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_76588

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ё
§
D__inference_conv2d_34_layer_call_and_return_conditional_losses_77219

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ll
X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         ll
i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         ll
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         nn
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         nn

 
_user_specified_nameinputs
╔
Ќ
(__inference_dense_24_layer_call_fn_77269

inputs
unknown:
║█

	unknown_0:

identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_76705o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ║█: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ║█
 
_user_specified_nameinputs
Е(
Ї
H__inference_sequential_14_layer_call_and_return_conditional_losses_76938
conv2d_32_input)
conv2d_32_76908:

conv2d_32_76910:
)
conv2d_33_76914:


conv2d_33_76916:
)
conv2d_34_76920:


conv2d_34_76922:
)
conv2d_35_76925:


conv2d_35_76927:
"
dense_24_76932:
║█

dense_24_76934:

identityѕб!conv2d_32/StatefulPartitionedCallб!conv2d_33/StatefulPartitionedCallб!conv2d_34/StatefulPartitionedCallб!conv2d_35/StatefulPartitionedCallб dense_24/StatefulPartitionedCallЄ
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallconv2d_32_inputconv2d_32_76908conv2d_32_76910*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_76620­
activation_1/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         яя
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_76631Ю
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0conv2d_33_76914conv2d_33_76916*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ▄▄
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_76644Ш
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         nn
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_76588Ъ
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_34_76920conv2d_34_76922*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_76662а
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_76925conv2d_35_76927*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         jj
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_76679Ш
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         55
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *T
fORM
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_76600с
flatten_14/PartitionedCallPartitionedCall)max_pooling2d_21/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ║█* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_flatten_14_layer_call_and_return_conditional_losses_76692Ї
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#flatten_14/PartitionedCall:output:0dense_24_76932dense_24_76934*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_dense_24_layer_call_and_return_conditional_losses_76705x
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
щ
NoOpNoOp"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЯЯ: : : : : : : : : : 2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:b ^
1
_output_shapes
:         ЯЯ
)
_user_specified_nameconv2d_32_input
з
c
G__inference_activation_1_layer_call_and_return_conditional_losses_76631

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:         яя
d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:         яя
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         яя
:Y U
1
_output_shapes
:         яя

 
_user_specified_nameinputs
Д

Ш
C__inference_dense_24_layer_call_and_return_conditional_losses_76705

inputs2
matmul_readvariableop_resource:
║█
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
║█
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ║█: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ║█
 
_user_specified_nameinputs
╔
a
E__inference_flatten_14_layer_call_and_return_conditional_losses_77260

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ║m  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ║█Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ║█"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         55
:W S
/
_output_shapes
:         55

 
_user_specified_nameinputs
ь
ъ
)__inference_conv2d_34_layer_call_fn_77208

inputs!
unknown:


	unknown_0:

identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ll
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_76662w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         ll
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         nn
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         nn

 
_user_specified_nameinputs
Д

Ш
C__inference_dense_24_layer_call_and_return_conditional_losses_77280

inputs2
matmul_readvariableop_resource:
║█
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
║█
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ║█: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ║█
 
_user_specified_nameinputs
ш
ъ
)__inference_conv2d_33_layer_call_fn_77178

inputs!
unknown:


	unknown_0:

identityѕбStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ▄▄
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_76644y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ▄▄
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         яя
: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         яя

 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┼
serving_default▒
U
conv2d_32_inputB
!serving_default_conv2d_32_input:0         ЯЯ<
dense_240
StatefulPartitionedCall:0         
tensorflow/serving/predict:Мр
П
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
П
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
 _jit_compiled_convolution_op"
_tf_keras_layer
Ц
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses"
_tf_keras_layer
П
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op"
_tf_keras_layer
Ц
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
П
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
П
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

@kernel
Abias
 B_jit_compiled_convolution_op"
_tf_keras_layer
Ц
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
╗
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias"
_tf_keras_layer
f
0
1
(2
)3
74
85
@6
A7
U8
V9"
trackable_list_wrapper
f
0
1
(2
)3
74
85
@6
A7
U8
V9"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ж
\trace_0
]trace_1
^trace_2
_trace_32■
-__inference_sequential_14_layer_call_fn_76735
-__inference_sequential_14_layer_call_fn_77029
-__inference_sequential_14_layer_call_fn_77054
-__inference_sequential_14_layer_call_fn_76905┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z\trace_0z]trace_1z^trace_2z_trace_3
Н
`trace_0
atrace_1
btrace_2
ctrace_32Ж
H__inference_sequential_14_layer_call_and_return_conditional_losses_77097
H__inference_sequential_14_layer_call_and_return_conditional_losses_77140
H__inference_sequential_14_layer_call_and_return_conditional_losses_76938
H__inference_sequential_14_layer_call_and_return_conditional_losses_76971┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z`trace_0zatrace_1zbtrace_2zctrace_3
МBл
 __inference__wrapped_model_76579conv2d_32_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Џ
diter

ebeta_1

fbeta_2
	gdecay
hlearning_ratem┤mх(mХ)mи7mИ8m╣@m║Am╗Um╝VmйvЙv┐(v└)v┴7v┬8v├@v─Av┼UvкVvК"
	optimizer
,
iserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ь
otrace_02л
)__inference_conv2d_32_layer_call_fn_77149б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zotrace_0
ѕ
ptrace_02в
D__inference_conv2d_32_layer_call_and_return_conditional_losses_77159б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zptrace_0
*:(
2conv2d_32/kernel
:
2conv2d_32/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
­
vtrace_02М
,__inference_activation_1_layer_call_fn_77164б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zvtrace_0
І
wtrace_02Ь
G__inference_activation_1_layer_call_and_return_conditional_losses_77169б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zwtrace_0
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ь
}trace_02л
)__inference_conv2d_33_layer_call_fn_77178б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z}trace_0
ѕ
~trace_02в
D__inference_conv2d_33_layer_call_and_return_conditional_losses_77189б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z~trace_0
*:(

2conv2d_33/kernel
:
2conv2d_33/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▒
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
Ш
ёtrace_02О
0__inference_max_pooling2d_20_layer_call_fn_77194б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zёtrace_0
Љ
Ёtrace_02Ы
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_77199б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЁtrace_0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
єnon_trainable_variables
Єlayers
ѕmetrics
 Ѕlayer_regularization_losses
іlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
№
Іtrace_02л
)__inference_conv2d_34_layer_call_fn_77208б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zІtrace_0
і
їtrace_02в
D__inference_conv2d_34_layer_call_and_return_conditional_losses_77219б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zїtrace_0
*:(

2conv2d_34/kernel
:
2conv2d_34/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Їnon_trainable_variables
јlayers
Јmetrics
 љlayer_regularization_losses
Љlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
№
њtrace_02л
)__inference_conv2d_35_layer_call_fn_77228б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zњtrace_0
і
Њtrace_02в
D__inference_conv2d_35_layer_call_and_return_conditional_losses_77239б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЊtrace_0
*:(

2conv2d_35/kernel
:
2conv2d_35/bias
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ћnon_trainable_variables
Ћlayers
ќmetrics
 Ќlayer_regularization_losses
ўlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ш
Ўtrace_02О
0__inference_max_pooling2d_21_layer_call_fn_77244б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЎtrace_0
Љ
џtrace_02Ы
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_77249б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Џnon_trainable_variables
юlayers
Юmetrics
 ъlayer_regularization_losses
Ъlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
­
аtrace_02Л
*__inference_flatten_14_layer_call_fn_77254б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zаtrace_0
І
Аtrace_02В
E__inference_flatten_14_layer_call_and_return_conditional_losses_77260б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zАtrace_0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
Ь
Дtrace_02¤
(__inference_dense_24_layer_call_fn_77269б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zДtrace_0
Ѕ
еtrace_02Ж
C__inference_dense_24_layer_call_and_return_conditional_losses_77280б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zеtrace_0
#:!
║█
2dense_24/kernel
:
2dense_24/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
0
Е0
ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЄBё
-__inference_sequential_14_layer_call_fn_76735conv2d_32_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
-__inference_sequential_14_layer_call_fn_77029inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
-__inference_sequential_14_layer_call_fn_77054inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЄBё
-__inference_sequential_14_layer_call_fn_76905conv2d_32_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
H__inference_sequential_14_layer_call_and_return_conditional_losses_77097inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЎBќ
H__inference_sequential_14_layer_call_and_return_conditional_losses_77140inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
бBЪ
H__inference_sequential_14_layer_call_and_return_conditional_losses_76938conv2d_32_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
бBЪ
H__inference_sequential_14_layer_call_and_return_conditional_losses_76971conv2d_32_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
мB¤
#__inference_signature_wrapper_77004conv2d_32_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ПB┌
)__inference_conv2d_32_layer_call_fn_77149inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_32_layer_call_and_return_conditional_losses_77159inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBП
,__inference_activation_1_layer_call_fn_77164inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
G__inference_activation_1_layer_call_and_return_conditional_losses_77169inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ПB┌
)__inference_conv2d_33_layer_call_fn_77178inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_33_layer_call_and_return_conditional_losses_77189inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_max_pooling2d_20_layer_call_fn_77194inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_77199inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ПB┌
)__inference_conv2d_34_layer_call_fn_77208inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_34_layer_call_and_return_conditional_losses_77219inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ПB┌
)__inference_conv2d_35_layer_call_fn_77228inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
D__inference_conv2d_35_layer_call_and_return_conditional_losses_77239inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBр
0__inference_max_pooling2d_21_layer_call_fn_77244inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 BЧ
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_77249inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
яB█
*__inference_flatten_14_layer_call_fn_77254inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щBШ
E__inference_flatten_14_layer_call_and_return_conditional_losses_77260inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
▄B┘
(__inference_dense_24_layer_call_fn_77269inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_dense_24_layer_call_and_return_conditional_losses_77280inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
Ф	variables
г	keras_api

Гtotal

«count"
_tf_keras_metric
c
»	variables
░	keras_api

▒total

▓count
│
_fn_kwargs"
_tf_keras_metric
0
Г0
«1"
trackable_list_wrapper
.
Ф	variables"
_generic_user_object
:  (2total
:  (2count
0
▒0
▓1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-
2Adam/conv2d_32/kernel/m
!:
2Adam/conv2d_32/bias/m
/:-

2Adam/conv2d_33/kernel/m
!:
2Adam/conv2d_33/bias/m
/:-

2Adam/conv2d_34/kernel/m
!:
2Adam/conv2d_34/bias/m
/:-

2Adam/conv2d_35/kernel/m
!:
2Adam/conv2d_35/bias/m
(:&
║█
2Adam/dense_24/kernel/m
 :
2Adam/dense_24/bias/m
/:-
2Adam/conv2d_32/kernel/v
!:
2Adam/conv2d_32/bias/v
/:-

2Adam/conv2d_33/kernel/v
!:
2Adam/conv2d_33/bias/v
/:-

2Adam/conv2d_34/kernel/v
!:
2Adam/conv2d_34/bias/v
/:-

2Adam/conv2d_35/kernel/v
!:
2Adam/conv2d_35/bias/v
(:&
║█
2Adam/dense_24/kernel/v
 :
2Adam/dense_24/bias/vф
 __inference__wrapped_model_76579Ё
()78@AUVBб?
8б5
3і0
conv2d_32_input         ЯЯ
ф "3ф0
.
dense_24"і
dense_24         
и
G__inference_activation_1_layer_call_and_return_conditional_losses_77169l9б6
/б,
*і'
inputs         яя

ф "/б,
%і"
0         яя

џ Ј
,__inference_activation_1_layer_call_fn_77164_9б6
/б,
*і'
inputs         яя

ф ""і         яя
И
D__inference_conv2d_32_layer_call_and_return_conditional_losses_77159p9б6
/б,
*і'
inputs         ЯЯ
ф "/б,
%і"
0         яя

џ љ
)__inference_conv2d_32_layer_call_fn_77149c9б6
/б,
*і'
inputs         ЯЯ
ф ""і         яя
И
D__inference_conv2d_33_layer_call_and_return_conditional_losses_77189p()9б6
/б,
*і'
inputs         яя

ф "/б,
%і"
0         ▄▄

џ љ
)__inference_conv2d_33_layer_call_fn_77178c()9б6
/б,
*і'
inputs         яя

ф ""і         ▄▄
┤
D__inference_conv2d_34_layer_call_and_return_conditional_losses_77219l787б4
-б*
(і%
inputs         nn

ф "-б*
#і 
0         ll

џ ї
)__inference_conv2d_34_layer_call_fn_77208_787б4
-б*
(і%
inputs         nn

ф " і         ll
┤
D__inference_conv2d_35_layer_call_and_return_conditional_losses_77239l@A7б4
-б*
(і%
inputs         ll

ф "-б*
#і 
0         jj

џ ї
)__inference_conv2d_35_layer_call_fn_77228_@A7б4
-б*
(і%
inputs         ll

ф " і         jj
Ц
C__inference_dense_24_layer_call_and_return_conditional_losses_77280^UV1б.
'б$
"і
inputs         ║█
ф "%б"
і
0         

џ }
(__inference_dense_24_layer_call_fn_77269QUV1б.
'б$
"і
inputs         ║█
ф "і         
Ф
E__inference_flatten_14_layer_call_and_return_conditional_losses_77260b7б4
-б*
(і%
inputs         55

ф "'б$
і
0         ║█
џ Ѓ
*__inference_flatten_14_layer_call_fn_77254U7б4
-б*
(і%
inputs         55

ф "і         ║█Ь
K__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_77199ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_20_layer_call_fn_77194ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ь
K__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_77249ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ к
0__inference_max_pooling2d_21_layer_call_fn_77244ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╦
H__inference_sequential_14_layer_call_and_return_conditional_losses_76938
()78@AUVJбG
@б=
3і0
conv2d_32_input         ЯЯ
p 

 
ф "%б"
і
0         

џ ╦
H__inference_sequential_14_layer_call_and_return_conditional_losses_76971
()78@AUVJбG
@б=
3і0
conv2d_32_input         ЯЯ
p

 
ф "%б"
і
0         

џ ┬
H__inference_sequential_14_layer_call_and_return_conditional_losses_77097v
()78@AUVAб>
7б4
*і'
inputs         ЯЯ
p 

 
ф "%б"
і
0         

џ ┬
H__inference_sequential_14_layer_call_and_return_conditional_losses_77140v
()78@AUVAб>
7б4
*і'
inputs         ЯЯ
p

 
ф "%б"
і
0         

џ Б
-__inference_sequential_14_layer_call_fn_76735r
()78@AUVJбG
@б=
3і0
conv2d_32_input         ЯЯ
p 

 
ф "і         
Б
-__inference_sequential_14_layer_call_fn_76905r
()78@AUVJбG
@б=
3і0
conv2d_32_input         ЯЯ
p

 
ф "і         
џ
-__inference_sequential_14_layer_call_fn_77029i
()78@AUVAб>
7б4
*і'
inputs         ЯЯ
p 

 
ф "і         
џ
-__inference_sequential_14_layer_call_fn_77054i
()78@AUVAб>
7б4
*і'
inputs         ЯЯ
p

 
ф "і         
└
#__inference_signature_wrapper_77004ў
()78@AUVUбR
б 
KфH
F
conv2d_32_input3і0
conv2d_32_input         ЯЯ"3ф0
.
dense_24"і
dense_24         
