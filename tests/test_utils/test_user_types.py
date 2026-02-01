import pytest
from cbfkit.utils.user_types import CertificateCollection, EMPTY_CERTIFICATE_COLLECTION
from cbfkit.certificates.packager import concatenate_certificates

def test_certificate_collection_addition():
    c1 = CertificateCollection([1], [2], [3], [4], [5])
    c2 = CertificateCollection([6], [7], [8], [9], [10])

    c3 = c1 + c2

    assert isinstance(c3, CertificateCollection)
    assert c3.functions == [1, 6]
    assert c3.jacobians == [2, 7]
    assert c3.hessians == [3, 8]
    assert c3.partials == [4, 9]
    assert c3.conditions == [5, 10]

def test_certificate_collection_sum():
    c1 = CertificateCollection([1], [], [], [], [])
    c2 = CertificateCollection([2], [], [], [], [])
    c3 = CertificateCollection([3], [], [], [], [])

    # Using sum
    c_sum = sum([c1, c2, c3], EMPTY_CERTIFICATE_COLLECTION)

    assert isinstance(c_sum, CertificateCollection)
    assert c_sum.functions == [1, 2, 3]

def test_concatenate_certificates_compatibility():
    c1 = CertificateCollection([1], [], [], [], [])
    c2 = CertificateCollection([2], [], [], [], [])

    c_concat = concatenate_certificates(c1, c2)

    assert isinstance(c_concat, CertificateCollection)
    assert c_concat.functions == [1, 2]

def test_addition_invalid_type():
    c1 = CertificateCollection([], [], [], [], [])
    with pytest.raises(TypeError):
        _ = c1 + "invalid"
