# -*- coding: utf-8 -*-
from typing import Tuple

import torch
from pytest_cases import parametrize_with_cases

from climsr.metrics.regression_accuracy import RegressionAccuracy

shape = (3, 128, 128)


def case_eps_zero_point_one_should_return_zero_percent_accuracy() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 0.1
    preds = torch.zeros(shape)
    targets = torch.ones(shape)
    expected_result = 0.0

    return eps, preds, targets, expected_result


def case_eps_zero_point_one_should_return_hundred_percent_accuracy_1() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 0.1
    preds = torch.ones(shape)
    targets = torch.ones(shape)
    expected_result = 1.0

    return eps, preds, targets, expected_result


def case_eps_zero_point_one_should_return_hundred_percent_accuracy_2() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 0.1
    preds = torch.ones(shape)
    rand = torch.rand(shape) / 100
    preds = preds - rand
    targets = torch.ones(shape)
    expected_result = 1.0

    return eps, preds, targets, expected_result


def case_eps_one_should_return_zero_percent_accuracy() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 1.0
    preds = torch.zeros(shape)
    targets = torch.ones(shape) + 1
    expected_result = 0.0

    return eps, preds, targets, expected_result


def case_eps_one_should_return_hundred_percent_accuracy_1() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 1.0
    preds = torch.ones(shape)
    targets = torch.ones(shape)
    expected_result = 1.0

    return eps, preds, targets, expected_result


def case_eps_one_should_return_hundred_percent_accuracy_2() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 1.0
    preds = torch.ones(shape)
    rand = torch.rand(shape) / 100
    preds = preds - rand
    targets = torch.ones(shape)
    expected_result = 1.0

    return eps, preds, targets, expected_result


def case_eps_zero_point_twenty_five_should_return_zero_percent_accuracy() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 0.25
    preds = torch.zeros(shape)
    targets = torch.ones(shape)
    expected_result = 0.0

    return eps, preds, targets, expected_result


def case_eps_zero_point_twenty_five_should_return_hundred_percent_accuracy_1() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 0.25
    preds = torch.ones(shape)
    targets = torch.ones(shape)
    expected_result = 1.0

    return eps, preds, targets, expected_result


def case_eps_zero_point_twenty_five_should_return_hundred_percent_accuracy_2() -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    eps = 0.25
    preds = torch.ones(shape)
    rand = torch.rand(shape) / 100
    preds = preds - rand
    targets = torch.ones(shape)
    expected_result = 1.0

    return eps, preds, targets, expected_result


@parametrize_with_cases("eps,preds,targets,expected_score", cases=".")
def test_should_work_properly(eps: float, preds: torch.Tensor, targets: torch.Tensor, expected_score: float) -> None:
    # arrange
    sut = RegressionAccuracy(eps=eps)

    # act
    acc = sut(preds, targets)

    # assert
    assert acc == expected_score
