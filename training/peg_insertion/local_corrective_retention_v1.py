"""Local corrective retention constraint V1 for the existing C+G actor."""
import chex
import jax.numpy as jnp
from training.peg_insertion.combined_q_anchor_ablation import CombinedQAnchorSACAgent
class LocalCorrectiveRetentionV1Agent(CombinedQAnchorSACAgent):
 def policy_loss_fn(self,batch,params,rng):
  rl_loss,info=super().policy_loss_fn(batch,params,rng)
  obs=batch['retention_observations'];target=batch['retention_actions']
  pred=self.forward_policy(obs,rng=rng,grad_params=params).mode();chex.assert_equal_shape([pred,target])
  retention=jnp.mean(jnp.square(pred[:,:6]-target[:,:6]));weighted=self.config['local_retention_lambda']*retention;total=rl_loss+weighted
  info=dict(info);info.update(rl_actor_loss=rl_loss,local_retention_loss=retention,weighted_local_retention_loss=weighted,actor_loss=total)
  return total,info
def with_local_retention_v1(agent,weight):
 return LocalCorrectiveRetentionV1Agent(state=agent.state,config={**agent.config,'local_retention_lambda':float(weight)})
