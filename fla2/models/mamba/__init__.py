from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla2.models.mamba.configuration_mamba import MambaConfig
from fla2.models.mamba.modeling_mamba import MambaForCausalLM, MambaModel

AutoConfig.register(MambaConfig.model_type, MambaConfig, exist_ok=True)
AutoModel.register(MambaConfig, MambaModel, exist_ok=True)
AutoModelForCausalLM.register(MambaConfig, MambaForCausalLM, exist_ok=True)


__all__ = ['MambaConfig', 'MambaForCausalLM', 'MambaModel']
