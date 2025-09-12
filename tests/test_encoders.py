import pytest
from mypackage import SMILESEncoder

def test_encode_maccs():
    encoder = SMILESEncoder()
    arr = encoder.encode_maccs("CCO")  # ethanol
    assert arr.shape[0] > 0

