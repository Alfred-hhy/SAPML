pub mod multi;
pub mod two;

pub use two::MpcTwoNet;
pub use multi::MpcMultiNet;

pub struct Stats {
    pub bytes_sent: usize,
    pub bytes_recv: usize,
    pub broadcasts: usize,
    pub to_king: usize,
    pub from_king: usize,
}

impl std::default::Default for Stats {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_recv: 0,
            broadcasts: 0,
            to_king: 0,
            from_king: 0,
        }
    }
}

pub trait MpcNet {

    fn am_king() -> bool {
        Self::party_id() == 0
    }
    fn n_parties() -> usize;
    fn party_id() -> usize;
    fn init_from_file(path: &str, party_id: usize);
    fn is_init() -> bool;
    fn deinit();
    fn reset_stats();
    fn stats() -> Stats;
    fn broadcast_bytes(bytes: &[u8]) -> Vec<Vec<u8>>;
    fn broadcast_bytes_unequal(bytes: &[u8]) -> Vec<Vec<u8>>;
    fn send_bytes_to_king(bytes: &[u8]) -> Option<Vec<Vec<u8>>>;
    fn recv_bytes_from_king(bytes: Option<Vec<Vec<u8>>>) -> Vec<u8>;

    fn compute_global_data_sent() -> usize;

    fn king_compute(bytes: &[u8], f: impl Fn(Vec<Vec<u8>>) -> Vec<Vec<u8>>) -> Vec<u8> {
        let king_response = Self::send_bytes_to_king(bytes).map(f);
        Self::recv_bytes_from_king(king_response)
    }
}
