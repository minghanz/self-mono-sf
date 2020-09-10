"""We wrote these customized collate functions to be compatible to C3D related inputs (CamInfo, etc. )"""
import torch
import re
from torch._six import container_abcs, string_classes, int_classes

import c3d

np_str_obj_array_pattern = re.compile(r'[SaUO]')

c3d_CamOpsType = (c3d.utils.cam.CamCrop, c3d.utils.cam.CamScale, c3d.utils.cam.CamFlip)

def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def default_collate_with_caminfo(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    ########### modification to default_collate compared with original in PyTorch
    ### dedicated collate function for CamInfo
    if isinstance(elem, c3d.utils.CamInfo):
        return c3d.utils.batch_cam_infos(batch)
    #############################################################################

    elif isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate_with_caminfo([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate_with_caminfo([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_with_caminfo(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate_with_caminfo(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def default_collate_with_camops_caminfo(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    ########### modification to default_collate compared with original in PyTorch
    ### dedicated collate function for c3d_CamOpsType
    try:
        if isinstance(elem, container_abcs.Sequence) and len(elem) > 0 and isinstance(elem[0], c3d_CamOpsType):
            return batch
        elif isinstance(elem, container_abcs.Sequence) and len(elem) == 0:
            return batch
        elif isinstance(elem, c3d.utils.CamInfo):
            return c3d.utils.batch_cam_infos(batch)
        #############################################################################

        elif isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                return default_collate_with_camops_caminfo([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            # timer = c3d.utils_general.Timing()
            # timer.log_temp("collate", True)
            # new_dict = {key: default_collate_with_camops_caminfo([d[key] for d in batch]) for key in elem}
            # timer.log_temp_end("collate", True)
            # return new_dict
            return {key: default_collate_with_camops_caminfo([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(default_collate_with_camops_caminfo(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [default_collate_with_camops_caminfo(samples) for samples in transposed]
    except Exception as e: 
        print("Some error happened:", e)
        # print("batch", batch)
        # print("elem", elem)
        print("elem_type", elem_type)
        
    raise TypeError(default_collate_err_msg_format.format(elem_type))

def decollate_with_camops_caminfo(elem):
    """go from a batched dict with batch_size=1 to an unbatched one. """

    assert isinstance(elem, container_abcs.Mapping)

    new_dict = {}
    for key, value in elem.items():
        if isinstance(value, list):
            new_dict[key] = value[0]
        elif isinstance(value, torch.Tensor):
            new_dict[key] = value[0]
        elif isinstance(value, c3d.utils.CamInfo):
            new_dict[key] = value
        else:
            raise NotImplementedError("unexpected value type for {}: {}".format(key, type(value)))
    return new_dict