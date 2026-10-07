"""One production synchronization policy; no algorithm or placement changes."""
POLICY = 'PER_LEAF_BLOCK_UNTIL_READY'
EXACT_ROOT_CAUSE = 'UNRESOLVED'

def block_until_ready(tree):
    import jax
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()
    return tree

class JaxView:
    def __init__(self, module):self.module=module
    def __getattr__(self,name):
        return block_until_ready if name=='block_until_ready' else getattr(self.module,name)
