use lazy_static::lazy_static;
use log::debug;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::Duration;
use std::convert::TryInto;
use std::thread::sleep;

use ark_std::{end_timer, start_timer};

use super::{MpcNet, Stats};

lazy_static! {
    static ref CONNECTIONS: Mutex<Connections> = Mutex::new(Connections::default());
}

macro_rules! get_ch {
    () => {
        CONNECTIONS.lock().expect("Poisoned FieldChannel")
    };
}

struct Peer {
    id: usize,
    addr: SocketAddr,
    stream: Option<TcpStream>,
}

struct Connections {
    id: usize,
    peers: Vec<Peer>,
    stats: Stats,
}

impl std::default::Default for Peer {
    fn default() -> Self {
        Self {
            id: 0,
            addr: "127.0.0.1:8000".parse().unwrap(),
            stream: None,
        }
    }
}

impl Connections {
    fn init_from_path(&mut self, path: &str, id: usize) {

        let f = BufReader::new(File::open(path).expect("host configuration path"));
        let mut peer_id = 0;
        for line in f.lines() {
            let line = line.unwrap();
            let trimmed = line.trim();
            if trimmed.len() > 0 {
                let addr: SocketAddr = trimmed
                    .parse()
                    .unwrap_or_else(|e| panic!("bad socket address: {}:\n{}", trimmed, e));
                let peer = Peer {
                    id: peer_id,
                    addr,
                    stream: None,
                };
                self.peers.push(peer);
                peer_id += 1;
            }
        }
        assert!(id < self.peers.len());
        self.id = id;
    }
    fn connect_to_all(&mut self) {
        let n = self.peers.len();
        for from_id in 0..n {
            for to_id in (from_id + 1)..n {
                debug!("{} to {}", from_id, to_id);
                if self.id == from_id {
                    let to_addr = self.peers[to_id].addr;
                    debug!("Contacting {}", to_id);
                    let stream = loop {
                        let mut ms_waited = 0;
                        match TcpStream::connect_timeout(&to_addr, Duration::from_secs(30)) {
                            Ok(s) => break s,
                            Err(e) => match e.kind() {
                                std::io::ErrorKind::ConnectionRefused
                                | std::io::ErrorKind::ConnectionReset => {
                                    ms_waited += 10;
                                    std::thread::sleep(std::time::Duration::from_millis(10));
                                    if ms_waited % 3_000 == 0 {
                                        debug!("Still waiting");
                                    } else if ms_waited > 30_000 {
                                        panic!("Could not find peer in 30s");
                                    }
                                }
                                _ => {
                                    panic!("Error during FieldChannel::new: {}", e);
                                }
                            },
                        }
                    };
                    stream.set_nodelay(true).unwrap();
                    self.peers[to_id].stream = Some(stream);
                } else if self.id == to_id {
                    debug!("Awaiting {}", from_id);
                    let listener = TcpListener::bind(self.peers[self.id].addr).unwrap();
                    let (stream, _addr) = listener.accept().unwrap();
                    stream.set_nodelay(true).unwrap();
                    self.peers[from_id].stream = Some(stream);
                }
            }
            debug!("Set up all connections");
            if from_id + 1 < n {
                if self.id == from_id {
                    self.peers[self.id + 1]
                        .stream
                        .as_mut()
                        .unwrap()
                        .write_all(&[0u8])
                        .unwrap();
                } else if self.id == from_id + 1 {
                    self.peers[self.id - 1]
                        .stream
                        .as_mut()
                        .unwrap()
                        .read_exact(&mut [0u8])
                        .unwrap();
                }
            }
        }
        debug!("Received note");
        let from_all = self.send_to_king(&[self.id as u8]);
        self.recv_from_king(from_all);
        for id in 0..n {
            if id != self.id {
                assert!(self.peers[id].stream.is_some());
            }
        }
    }
    fn am_king(&self) -> bool {
        self.id == 0
    }
    fn broadcast(&mut self, bytes_out: &[u8]) -> Vec<Vec<u8>> {
        let m = bytes_out.len();
        let own_id = self.id;
        self.stats.bytes_sent += (self.peers.len() - 1) * m;
        self.stats.bytes_recv += (self.peers.len() - 1) * m;
        self.stats.broadcasts += 1;
        let r = self
            .peers
            .par_iter_mut()
            .enumerate()
            .map(|(id, peer)| {
                let mut bytes_in = vec![0u8; m];
                if id < own_id {
                    let stream = peer.stream.as_mut().unwrap();
                    stream.read_exact(&mut bytes_in[..]).unwrap();
                    stream.write_all(bytes_out).unwrap();
                } else if id == own_id {
                    bytes_in.copy_from_slice(bytes_out);
                } else {
                    let stream = peer.stream.as_mut().unwrap();
                    stream.write_all(bytes_out).unwrap();
                    stream.read_exact(&mut bytes_in[..]).unwrap();
                };
                bytes_in
            })
            .collect();
        r
    }
    fn send_to_king(&mut self, bytes_out: &[u8]) -> Option<Vec<Vec<u8>>> {
        let m = bytes_out.len();
        let own_id = self.id;
        self.stats.to_king += 1;
        let r = if self.am_king() {
            self.stats.bytes_recv += (self.peers.len() - 1) * m;
            Some(
                self.peers
                    .par_iter_mut()
                    .enumerate()
                    .map(|(id, peer)| {
                        let mut bytes_in = vec![0u8; m];
                        if id == own_id {
                            bytes_in.copy_from_slice(bytes_out);
                        } else {
                            let stream = peer.stream.as_mut().unwrap();
                            stream.read_exact(&mut bytes_in[..]).unwrap();
                        };
                        bytes_in
                    })
                    .collect(),
            )
        } else {
            self.stats.bytes_sent += m;
            self.peers[0]
                .stream
                .as_mut()
                .unwrap()
                .write_all(bytes_out)
                .unwrap();
            None
        };
        r
    }
    fn recv_from_king(&mut self, bytes_out: Option<Vec<Vec<u8>>>) -> Vec<u8> {
        let own_id = self.id;
        self.stats.from_king += 1;
        if self.am_king() {
            let bytes_out = bytes_out.unwrap();
            let m = bytes_out[0].len();
            let bytes_size = (m as u64).to_le_bytes();
            self.stats.bytes_sent += (self.peers.len() - 1) * (m + 8);
            self.peers
                .par_iter_mut()
                .enumerate()
                .filter(|p| p.0 != own_id)
                .for_each(|(id, peer)| {
                    let stream = peer.stream.as_mut().unwrap();
                    assert_eq!(bytes_out[id].len(), m);
                    stream.write_all(&bytes_size).unwrap();
                    stream.write_all(&bytes_out[id]).unwrap();
                });
            bytes_out[own_id].clone()
        } else {
            let stream = self.peers[0].stream.as_mut().unwrap();
            let mut bytes_size = [0u8; 8];
            stream.read_exact(&mut bytes_size).unwrap();
            let m = u64::from_le_bytes(bytes_size) as usize;
            self.stats.bytes_recv += m;
            let mut bytes_in = vec![0u8; m];
            stream.read_exact(&mut bytes_in).unwrap();
            bytes_in
        }
    }
    fn broadcast_unequal_lengths(&mut self, bytes_out: &[u8]) -> Vec<Vec<u8>> {
        let m = bytes_out.len();
        let own_id = self.id;
        let sizes: Vec<usize> = self.broadcast(m.to_le_bytes().as_slice())
            .into_iter().map(|b| usize::from_le_bytes(b.try_into().unwrap()))
            .collect();

        self.stats.bytes_sent += (self.peers.len() - 1) * m;
        self.stats.bytes_recv += sizes.iter().enumerate().filter(|(i, _)| *i != own_id).map(|(_, s)| *s).sum::<usize>();
        self.stats.broadcasts += 1;
        let r = self
            .peers
            .par_iter_mut()
            .zip(sizes.into_par_iter())
            .enumerate()
            .map(|(id, (peer, incoming_size))| {
                let mut bytes_in = vec![0u8; incoming_size];
                if id < own_id {
                    let stream = peer.stream.as_mut().unwrap();
                    stream.read_exact(&mut bytes_in[..]).unwrap();
                    stream.write_all(bytes_out).unwrap();
                } else if id == own_id {
                    bytes_in.copy_from_slice(bytes_out);
                } else {
                    let stream = peer.stream.as_mut().unwrap();
                    stream.write_all(bytes_out).unwrap();
                    stream.read_exact(&mut bytes_in[..]).unwrap();
                };
                bytes_in
            })
            .collect();
        r
    }
    fn uninit(&mut self) {
        for p in &mut self.peers {
            p.stream = None;
        }
    }
    fn compute_global_data_sent(&mut self) -> usize {
        let bytes_sent = self.stats.bytes_sent.to_le_bytes();
        let result_all = self.broadcast(&bytes_sent)
            .into_iter().map(|b| usize::from_le_bytes(b.try_into().unwrap()))
            .sum();

     self.stats.bytes_sent -= (self.peers.len() - 1) * bytes_sent.len();
        self.stats.bytes_recv -= (self.peers.len() - 1) * bytes_sent.len();
        self.stats.broadcasts -= 1;

        return result_all;
    }
}

pub struct MpcMultiNet;

impl MpcNet for MpcMultiNet {
    fn party_id() -> usize {
        get_ch!().id
    }

    fn n_parties() -> usize {
        get_ch!().peers.len()
    }

    fn init_from_file(path: &str, party_id: usize) {
        let mut ch = get_ch!();
        ch.init_from_path(path, party_id);
        ch.connect_to_all();
    }

    fn is_init() -> bool {
        get_ch!()
            .peers
            .first()
            .map(|p| p.stream.is_some())
            .unwrap_or(false)
    }

    fn deinit() {
        get_ch!().uninit()
    }

    fn reset_stats() {
        get_ch!().stats = Stats::default();
    }

    fn stats() -> crate::Stats {
        get_ch!().stats.clone()
    }

    fn broadcast_bytes(bytes: &[u8]) -> Vec<Vec<u8>> {
        get_ch!().broadcast(bytes)
    }

    fn broadcast_bytes_unequal(bytes: &[u8]) -> Vec<Vec<u8>> {
        get_ch!().broadcast_unequal_lengths(bytes)
    }

    fn send_bytes_to_king(bytes: &[u8]) -> Option<Vec<Vec<u8>>> {
        get_ch!().send_to_king(bytes)
    }

    fn recv_bytes_from_king(bytes: Option<Vec<Vec<u8>>>) -> Vec<u8> {
        get_ch!().recv_from_king(bytes)
    }

    fn compute_global_data_sent() -> usize {
        get_ch!().compute_global_data_sent()
    }
}
