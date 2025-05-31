
import abc
import subprocess
import enum
import os
import shlex
from typing import Dict, List

def program_args_cmdline(program_args: Dict[str, str]) -> List[str]:
    if program_args is None:
        return []
    list_tuples = [(f'-{k}', v) for k, v in program_args.items()]

    flat_list = [item for sublist in list_tuples for item in sublist]
    return flat_list

def script_name_and_args_to_correct_execution_name(script_name, script_args):
    serialized_args = [f'{k}__{v}' for k, v in script_args.items()]
    return f"{'-'.join([script_name] + serialized_args)}"


class BaseRunner(abc.ABC):


    @property
    def program(self):
        return str(self._program())

    @abc.abstractmethod
    def _program(self):
        pass

    @property
    def args(self):
        args_filtered = filter(lambda x: len(x) > 0, self._args())
        return [str(s) for s in args_filtered]

    @abc.abstractmethod
    def _args(self):
        pass

    @property
    def env(self):
        return self._env()

    @abc.abstractmethod
    def _env(self):
        pass

    def run(self, stdout=None, stderr=None):
        subprocess.run(
            " ".join([self.program] + self.args),
            shell=True,
            cwd="./MP-SPDZ/",
            check=True,
            capture_output=False,
            env=self.env,
            stdout=stdout,
            stderr=stderr
        )


class CompilerArguments(enum.Enum):
    EMULATE_X = ['-R', '64']
    REPLICATED_RING_PARTY_X = ['-R', "64"]
    REP4_RING_PARTY_X = ['-R', "64", '-Z', '4', '-C']  #
    BRAIN_PARTY_X = ['-R', '64']
    REPLICATED_BIN_PARTY_X = ['-B', '64']
    PS_REP_BIN_PARTY_X = ['-B', '64']
    SHAMIR_PARTY_X = ["-F", "64"]
    MALICIOUS_SHAMIR_PARTY_X = ["-F", "64"]
    ATLAS_PARTY_X = ["-F", "64"]
    MAL_ATLAS_PARTY_X = ["-F", "64"]
    REP_FIELD_PARTY = ["-F", "64"]
    MAL_REP_FIELD_PARTY = ["-F", "64"]
    MAL_REP_RING_PARTY = ["-R", "64"]

    SY_REP_RING_PARTY = ['-R', "64"]
    SY_REP_FIELD_PARTY = ['-F', "64"]
    PS_REP_FIELD_PARTY = ['-F', "64"]
    SPDZ2K_PARTY = ['-R', "64"]
    SEMI2K_PARTY = ['-R', "64"]
    SEMI_PARTY = ['-F', "128"]
    MASCOT_PARTY = ['-F', "64"]
    MASCOT_OFFLINE = ['-F', "64"]

class CompilerRunner(BaseRunner):

    def __init__(self, script_name, script_args, compiler_args, code_dir):
        self._script_name = script_name
        self._script_args = script_args
        self._compiler_args = compiler_args
        self._code_dir = code_dir

    def _program(self):
        return "./compile.py"

    def _env(self):
        my_env = os.environ.copy()
        if "PYTHONPATH" in my_env.keys():
            my_env["PYTHONPATH"] = f"{my_env['PYTHONPATH']}:{os.path.join(self._code_dir, 'scripts/')}"
        else:
            my_env["PYTHONPATH"] = f"{os.path.join(self._code_dir,'scripts/')}"
        return my_env


    def _args(self):
        serialized_args = [f'{k}__{v}' for k, v in self._script_args.items()]
        return self._compiler_args + \
            [os.path.join(self._code_dir, "scripts", f"{self._script_name}.mpc")] \
             + serialized_args


class ScriptBaseRunner(BaseRunner):
    def __init__(self, output_prefix, script_name, args, player_0_host, player_id, custom_prime, custom_prime_length, player_count, program_args):
        self.output_prefix = output_prefix
        self.script_name = script_name
        self.script_args = args
        self.player_0_host = player_0_host
        self.player_id = player_id
        self.player_count = player_count
        self.custom_prime = custom_prime
        self.custom_prime_length = custom_prime_length
        assert not (self.custom_prime is not None and self.custom_prime_length is not None),\
            "It is not possible to specify a custom prime AND a custom prime length!"
        self.program_args = program_args

    def _env(self):
        my_env = os.environ.copy()
        return my_env


class EmulatorRunner(ScriptBaseRunner):
    def _program(self):
        return "./emulate.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]



class ReplicatedRingPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run ReplicatedRingPartyRunner")
        return "./replicated-ring-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}"] + program_args_flat + [
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
            ]


class Replicated4RingPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run Replicated4RingPartyRunner")
        return "./rep4-ring-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]


class BrainPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run  BrainPartyRunner")
        return "./brain-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]


class ReplicatedBinPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run ReplicatedBinPartyRunner")
        return "./replicated-bin-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class PsReplicatedBinPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run PsReplicatedBinPartyRunner")
        return "./ps-rep-bin-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class SyReplicatedRingPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run SyReplicatedRingPartyRunner")
        return "./sy-rep-ring-party.x"

    def _args(self):
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class SyReplicatedFieldPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run SyReplicatedFieldPartyRunner")
        return "./sy-rep-field-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]


class PsReplicatedFieldPartyRunner(ScriptBaseRunner):
    def _program(self):
        print("Run PsReplicatedFieldPartyRunner")
        return "./ps-rep-field-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class ShamirPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./shamir-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
            custom_prime_arg, custom_prime_length_arg,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            "-N", f"{self.player_count}",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class MaliciousShamirPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./malicious-shamir-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
            "-h", f"{self.player_0_host}",
            "-pn", "12300",
                "-v",
            "-N", f"{self.player_count}",
            f"{self.player_id}",
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class AtlasPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./atlas-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                "-N", f"{self.player_count}",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class MaliciousAtlasPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./mal-atlas-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                "-N", f"{self.player_count}",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)]

class ReplicatedFieldPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./replicated-field-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class MaliciousReplicatedFieldPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./malicious-rep-field-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]

class MaliciousReplicatedRingPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./malicious-rep-ring-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-v",
                f"{self.player_id}"] + program_args_flat + [
            script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
        ]


class MascotPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./mascot-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class SemiPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./semi-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class LowgearPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./lowgear-party.x"

    def _args(self):
        program_args_flat = program_args_cmdline(self.program_args)
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return (["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}"]
                + program_args_flat + [
                    script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ])

class HighgearPartyRunner(ScriptBaseRunner):
    def _program(self):
        return "./highgear-party.x"

    def _args(self):
        custom_prime_arg = f"-P {self.custom_prime}" if self.custom_prime is not None else ""
        custom_prime_length_arg = f"-lgp {self.custom_prime_length}" if self.custom_prime is not None else ""
        return ["-OF", self.output_prefix,
                custom_prime_arg, custom_prime_length_arg,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                "-v",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class MascotOfflineRunner(ScriptBaseRunner):
    def _program(self):
        return "./mascot-offline.x"

    def _args(self):
        return ["-OF", self.output_prefix,
                "-h", f"{self.player_0_host}",
                "-pn", "12300",
                "-N", f"{self.player_count}",
                f"{self.player_id}",
                script_name_and_args_to_correct_execution_name(self.script_name, self.script_args)
                ]

class ProtocolRunners(enum.Enum):
    EMULATE_X = EmulatorRunner
    REPLICATED_RING_PARTY_X = ReplicatedRingPartyRunner
    REP4_RING_PARTY_X = Replicated4RingPartyRunner
    BRAIN_PARTY_X=BrainPartyRunner
    REPLICATED_BIN_PARTY_X=ReplicatedBinPartyRunner
    PS_REP_BIN_PARTY_X=PsReplicatedBinPartyRunner
    SHAMIR_PARTY_X=ShamirPartyRunner
    ATLAS_PARTY_X=AtlasPartyRunner
    MAL_ATLAS_PARTY_X=MaliciousAtlasPartyRunner
    MALICIOUS_SHAMIR_PARTY_X=MaliciousShamirPartyRunner
    REP_FIELD_PARTY=ReplicatedFieldPartyRunner
    MAL_REP_FIELD_PARTY=MaliciousReplicatedFieldPartyRunner
    MAL_REP_RING_PARTY=MaliciousReplicatedRingPartyRunner
    SY_REP_RING_PARTY=SyReplicatedRingPartyRunner
    SY_REP_FIELD_PARTY=SyReplicatedFieldPartyRunner
    PS_REP_FIELD_PARTY=PsReplicatedFieldPartyRunner
    MASCOT_PARTY=MascotPartyRunner
    SEMI_PARTY=SemiPartyRunner
    MASCOT_OFFLINE=MascotOfflineRunner
    LOWGEAR_PARTY=LowgearPartyRunner
    HIGHGEAR_PARTY=HighgearPartyRunner
