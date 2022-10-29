import torch
import torch.nn as nn

class ModelMonitor:
    
    def __init__(
        self,
        model: nn.Module,
        verbose: bool = True
    ):
        """Model Monitor init

        Args:
            model (nn.Module): model
            verbose (bool): if verbose
        """
        self.model = model
        self.verbose = verbose
        
        self.layer_found = False
        
        self.layer_data = {}
        self.layer_handle = {}
    
    def add_layer(
        self,
        layer_name: str,
        hook_type: str = "output",
        detach: bool = False
    ):
        """adds hook to a layer

        Args:
            layer_name (str): layer to hook
            hook_type (str, optional): which type of hook. Defaults to "output".
            detach (bool, optional): if detaching data. Defaults to False.

        Raises:
            ValueError: if layer is not found
        """
        self.layer_found = False
        self._explore_and_hook(
            model=self.model,
            root_layer_name="",
            target_layer=layer_name,
            hook_type=hook_type,
            detach=detach
        )
        
        if not self.layer_found:
            raise ValueError(f"{layer_name} not found in the model. Sure it exists??")
        
    def remove_layer(
        self,
        layer_name: str
    ):
        """removes layer's hook

        Args:
            layer_name (str): layer name

        Raises:
            ValueError: if layer is not found
        """
        if layer_name not in self.layer_data:
            raise ValueError(f"{layer_name} not found.")
        
        # removing hook
        self.layer_handle[layer_name].remove()
        del self.layer_handle[layer_name]
        del self.layer_data[layer_name]
    
    def get_data(
        self,
        layer_name: str
    ) -> torch.Tensor:
        """returns layer data

        Args:
            layer_name (str): layer name

        Raises:
            ValueError: if layer is not found.

        Returns:
            torch.Tensor: layer data
        """
        if layer_name not in self.layer_data:
            raise ValueError(f"{layer_name} not found.")
        return self.layer_data[layer_name]
         
    def _explore_and_hook(
        self,
        model: nn.Module,
        root_layer_name: str,
        target_layer: str,
        hook_type: str = "output",
        detach: bool = False
    ):
        """explores model structure and attach a hook

        Args:
            model (nn.Module): model
            root_layer_name (str): base layer name
            target_layer (str): target layer to find
            hook_type (str, optional): hook type (input/output). Defaults to "output".
            detach (bool, optional): if data to detach. Defaults to False.
        """
        
        for layer_name, layer in model.named_children():
            complete_name = root_layer_name + "." + layer_name
            if self.verbose:
                print(f"> {complete_name} -> {type(layer)}")
            
            if len(list(layer.children())) == 0:
                if complete_name == "." + target_layer:
                    print(f"[SUCCESS] Layer '{target_layer}' found!")
                    self._register_forward_hook(
                        layer=layer,
                        layer_name=target_layer,
                        hook_type=hook_type,
                        detach=detach
                    )
                    self.layer_found = True
                    return
                
            self._explore_and_hook(
                model=layer,
                root_layer_name=root_layer_name + "." + layer_name,
                target_layer=target_layer,
                hook_type=hook_type,
                detach=detach
            )
                             
    def _register_forward_hook(
        self,
        layer: nn.Module,
        layer_name: str,
        hook_type: str = "output",
        detach: bool = False
    ):
        """registers the hook for the layer

        Args:
            layer (nn.Module): layer
            layer_name (str): layer name
            hook_type (str, optional): type of hook (output/input). Defaults to "output".
            detach (bool, optional): if data to detach. Defaults to False.
        """
        
        assert hook_type in ["output", "input"], f"Hook type must be either input or output, not {hook_type}"
        
        def hook_output(layer, input, output):
            self.layer_data[layer_name] = output
            
        def hook_input(layer, input, output):
            # TODO: explore this
            self.layer_data[layer_name] = input[0]
          
        if hook_type == "output":
            self.layer_handle[layer_name] = layer.register_forward_hook(hook_output)
        else:
            self.layer_handle[layer_name] = layer.register_forward_hook(hook_input)