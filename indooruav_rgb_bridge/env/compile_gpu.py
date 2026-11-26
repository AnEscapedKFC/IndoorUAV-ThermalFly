"""Compile the inference graph from shapes to check device support before loading weights."""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi.models import pi0
cfg=pi0.Pi0Config(paligemma_variant="gemma_2b_lora",action_expert_variant="gemma_300m_lora",action_horizon=10,max_token_len=240)
m=nnx.eval_shape(cfg.create,jax.random.key(0))
graph,st=nnx.split(m)
def target(s):
 if isinstance(s,jax.ShapeDtypeStruct):return jax.ShapeDtypeStruct(s.shape,jnp.bfloat16 if jnp.issubdtype(s.dtype,jnp.floating) else s.dtype)
 return s
st=jax.tree.map(target,st)
def fun(state,*args,**kwargs):
 mod=nnx.merge(graph,state)
 return pi0.Pi0.sample_actions(mod,*args,**kwargs)
obs=cfg.inputs_spec()[0]
print('lowering graph',flush=True)
from openpi.shared.array_typing import disable_typechecking
with disable_typechecking():
 out=jax.jit(fun).lower(st,jax.random.key(0),obs).compile()
print('static GPU memory analysis',out.memory_analysis(),flush=True)
