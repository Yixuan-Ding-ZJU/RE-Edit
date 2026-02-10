"""
Base reward model class
Reward评分模型抽象基类
"""

from abc import abstractmethod
from typing import Any, Dict, Optional
from PIL import Image

from ..base import BaseModel


class BaseRewardModel(BaseModel):
    """
    Reward评分模型抽象基类
    
    所有具体的Reward模型实现都应该继承这个类，并实现score方法
    """
    
    @abstractmethod
    def score(self,
              edited_image: Image.Image,
              original_description: str,
              edit_instruction: str,
              system_prompt: str,
              user_prompt: str,
              original_image: Optional[Image.Image] = None,
              rationale: Optional[str] = None,
              **kwargs) -> float:
        """
        对编辑后的图像进行评分
        
        Args:
            edited_image: 编辑后的PIL图像
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            original_image: 原始图像（可选，某些模型可能需要）
            rationale: 编辑任务难点描述（可选）
            **kwargs: 其他参数
            
        Returns:
            评分（通常是0-10的浮点数）
        """
        pass
    
    def __call__(self,
                 edited_image: Image.Image,
                 original_description: str,
                 edit_instruction: str,
                 system_prompt: str,
                 user_prompt: str,
                 original_image: Optional[Image.Image] = None,
                 rationale: Optional[str] = None,
                 **kwargs) -> float:
        """
        调用接口，直接调用score
        
        Args:
            edited_image: 编辑后的PIL图像
            original_description: 原始图像描述
            edit_instruction: 编辑指令
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            original_image: 原始图像（可选）
            rationale: 编辑任务难点描述（可选）
            **kwargs: 其他参数
            
        Returns:
            评分
        """
        return self.score(
            edited_image=edited_image,
            original_description=original_description,
            edit_instruction=edit_instruction,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            original_image=original_image,
            rationale=rationale,
            **kwargs
        )
    
    def batch_score(self,
                   edited_images: list,
                   original_descriptions: list,
                   edit_instructions: list,
                   system_prompts: list,
                   user_prompts: list,
                   original_images: Optional[list] = None,
                   rationales: Optional[list] = None,
                   **kwargs) -> list:
        """
        批量评分（默认实现，可被子类覆盖以优化性能）
        
        Args:
            edited_images: 编辑后的图像列表
            original_descriptions: 原始图像描述列表
            edit_instructions: 编辑指令列表
            system_prompts: 系统prompt列表
            user_prompts: 用户prompt列表
            original_images: 原始图像列表（可选）
            rationales: 编辑任务难点描述列表（可选）
            **kwargs: 其他参数
            
        Returns:
            评分列表
        """
        n = len(edited_images)
        if not all(len(lst) == n for lst in [original_descriptions, edit_instructions, 
                                              system_prompts, user_prompts]):
            raise ValueError("All input lists must have the same length")
        
        if original_images is None:
            original_images = [None] * n
        
        if rationales is None:
            rationales = [None] * n
        
        scores = []
        for i in range(n):
            score = self.score(
                edited_image=edited_images[i],
                original_description=original_descriptions[i],
                edit_instruction=edit_instructions[i],
                system_prompt=system_prompts[i],
                user_prompt=user_prompts[i],
                original_image=original_images[i],
                rationale=rationales[i],
                **kwargs
            )
            scores.append(score)
        
        return scores


