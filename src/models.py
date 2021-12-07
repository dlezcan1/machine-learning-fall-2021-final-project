from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class LinearReLU( nn.Linear ):
    def __init__( self, in_features, out_features, bias: bool = True, device=None, dtype=None ):
        super().__init__( in_features, out_features, bias=bias, device=device, dtype=dtype )

    # __init__

    def forward( self, x: Tensor ) -> Tensor:
        x = super().forward( x )
        x = F.relu( x )

        return x

    # forward


# class: LinearReLU

class FeedForward( nn.Module ):
    def __init__( self ):
        super( FeedForward, self ).__init__()
        self.fc1 = nn.Linear( 100, 10 )
        self.fc2 = nn.Linear( 10, 1 )

    # __init__

    def forward( self, x ):
        x = self.fc1( x )
        x = F.relu( x )
        x = self.fc2( x )

        return x

    # forward


# class: FeedForward

class FeedForwardNLayers( nn.Module ):
    def __init__( self, layer_outs: list ):
        super().__init__()
        self.inp = LinearReLU( 100, layer_outs[ 0 ] )
        self.hidden = nn.Sequential(*[ LinearReLU( layer_outs[ i ], layer_outs[ i + 1 ] ) for i in
                        range( len( layer_outs ) - 1 ) ])
        self.out = nn.Linear( layer_outs[ -1 ], 1 )

    # __init__

    def forward( self, x ):
        x = self.inp( x )
        x = self.hidden( x )
        x = self.out( x )
        return x
    # forward
