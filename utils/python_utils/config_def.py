
import pydantic
import enum
import typing

class ProtocolChoices(enum.Enum):
    EMULATE_X = "emulate_env"
    REPLICATED_RING_PARTY_X = "semi_honest_3"
    REP4_RING_PARTY_X = "rep4-ring-party"
    BRAIN_PARTY_X = "malicious_3_party"
    REPLICATED_BIN_PARTY_X = "semi_honest_bin_3"
    PS_REP_BIN_PARTY_X = "malicious_bin_3"
    SHAMIR_PARTY_X = "shamir_semi_honest_n"
    MALICIOUS_SHAMIR_PARTY_X = "shamir_malicious_n"
    ATLAS_PARTY_X = "atlas_n"
    MAL_ATLAS_PARTY_X = "mal_atlas_n"
    REP_FIELD_PARTY = "rep-field-party"
    MAL_REP_FIELD_PARTY = "mal-rep-field-party"
    MAL_REP_RING_PARTY = "mal-rep-ring-party"

    SY_REP_RING_PARTY = "sy-rep-ring-party"
    SY_REP_FIELD_PARTY = "sy-rep-field-party"
    PS_REP_FIELD_PARTY = "ps-rep-field-party"
    SPDZ2K_PARTY = "spdz2k-party"
    SEMI2K_PARTY = "semi2k-party"
    SEMI_PARTY = "semi-party"
    MASCOT_PARTY = "mascot-party"
    MASCOT_OFFLINE = "mascot-offline"
    LOWGEAR_PARTY = "lowgear-party"
    HIGHGEAR_PARTY = "lowgear-party"


class ArgumentLineConfig(pydantic.BaseModel):

    player_id: int
    sleep_time: float

class JsoncMpcConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    script_name: str
    script_args: typing.Dict[str, object]
    protocol_setup: ProtocolChoices
    stage: typing.Union[typing.Literal['compile', 'run'], typing.List[typing.Literal['compile', 'run']]]
    custom_prime: typing.Optional[str] = None
    custom_prime_length: typing.Optional[str] = None

    compiler_args: list[str] = None
    program_args: typing.Dict[str, str] = None

    domain: typing.Optional[str] = None

class JsonConsistencyConfig(pydantic.BaseModel,extra=pydantic.Extra.forbid):
    hosts_file: str
    type: typing.Literal['pc', 'cerebro', 'sha3', 'sha3s']
    pc: typing.Literal['kzg', 'ipa', 'ped']
    abs_path_to_code_dir: str
    pp_args: int
    prover_party: typing.Optional[int] = None
    eval_point: typing.Optional[str] = None
    single_random_eval_point: bool = True
    gen_pp: bool = False
    use_split: bool = False


class JsonConfigModel(pydantic.BaseModel,extra=pydantic.Extra.ignore):
    mpc: JsoncMpcConfig
    consistency_args: typing.Optional[JsonConsistencyConfig] = None
    commit_output: typing.Optional[bool] = None
    convert_ring_bits: int = 34
    convert_n_threads: int = 18
    convert_chunk_size: int = 500000
    convert_debug: bool = False
    sleep_time: float = 5.0
    remove_input_files: bool = True

def parse_json_config(config_path):
    config_obj = JsonConfigModel.parse_file(config_path)
    return config_obj

def build_task_config(json_config_obj: JsonConfigModel, player_number: int,
                    result_dir: str):

    conf_obj = TaskConfig(
        player_id=player_number,
        sleep_time=json_config_obj.sleep_time,
        player_count=json_config_obj.mpc.player_count,
        player_0_hostname=json_config_obj.mpc.player_0_hostname,
        abs_path_to_code_dir=json_config_obj.mpc.abs_path_to_code_dir,
        protocol_setup=json_config_obj.mpc.protocol_setup,
        script_args=json_config_obj.mpc.script_args,
        script_name=json_config_obj.mpc.script_name,
        custom_prime=json_config_obj.mpc.custom_prime,
        custom_prime_length=json_config_obj.mpc.custom_prime_length,
        result_dir=result_dir,
        stage=json_config_obj.mpc.stage,
        compiler_args=json_config_obj.mpc.compiler_args,
        program_args=json_config_obj.mpc.program_args,
        consistency_args=json_config_obj.consistency_args,
        commit_output=json_config_obj.commit_output,
        convert_ring_bits=json_config_obj.convert_ring_bits,
        convert_n_threads=json_config_obj.convert_n_threads,
        convert_chunk_size=json_config_obj.convert_chunk_size,
        convert_debug=json_config_obj.convert_debug,
        remove_input_files=json_config_obj.remove_input_files
    )
    return conf_obj


class TaskConfig(pydantic.BaseModel):

    player_id: int
    sleep_time: float
    player_count: int
    player_0_hostname: str
    abs_path_to_code_dir: str
    # MPC specific options
    script_name: str
    script_args: typing.Dict[str, object]
    protocol_setup: ProtocolChoices
    result_dir: str
    stage: typing.Union[typing.Literal['compile', 'run'], typing.List[typing.Literal['compile', 'run']]]
    program_args: dict = None
    custom_prime: typing.Optional[str] = None
    custom_prime_length: typing.Optional[str] = None

    convert_ring_if_needed: bool = True
    convert_ring_bits: int
    convert_n_threads: int
    convert_chunk_size: int
    convert_debug: bool

    compiler_args: list = None
    consistency_args: typing.Optional[JsonConsistencyConfig] = None
    commit_output: typing.Optional[bool] = False

    remove_input_files: bool

    @pydantic.validator('stage')
    def convert_to_list(cls, v):
        if not isinstance(v, list):
            v = [v]
        return v