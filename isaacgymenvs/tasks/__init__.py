# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from .go1_dribble_traj import Go1DribblerTraj
from .go1_dribble import Go1Dribbler
from .go1_dribble_test import Go1DribblerTest
from .cassie_dribble import CaDribbler
from .nao_dribble import NaoDribbler
from .ant import Ant


# Mappings from strings to environments
isaacgym_task_map = {
    "Go1Dribble": Go1Dribbler,
    "Go1DribbleTest": Go1DribblerTest,
    "Go1DribbleTraj": Go1DribblerTraj,
    "NaoDribble": NaoDribbler,
    "CaDribble": CaDribbler,
    "Ant": Ant,
}


try:
    from .go1_ball_real import BallReal
    from .go1real import Go1Real
    from .go1_dribble_real import DribbleReal
except:
    print("No real robot envs and packages installed")
else:
    isaacgym_task_map.update(
        {
            "Go1BallReal": BallReal,
            "Go1Real": Go1Real,
            "Go1DribbleReal": DribbleReal,
        }
    )
