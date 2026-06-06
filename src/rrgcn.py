import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

from rgcn.layers import UnionRGCNLayer, RGAT, UnionRGCNLayer2, UnionRGCNLayer3, UnionRGATLayer, CompGCNLayer
from src.model import BaseRGCN
from src.decoder import ConvTransE, ConvTransR
from collections import defaultdict

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                            activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn" or self.encoder_name == "kbat" or self.encoder_name == "compgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')


class RGCNCell2(BaseRGCN):
    """Global subgraph encoder cell.

    In Module 3, after extracting the global candidate subgraph via RWR,
    this cell applies standard R-GCN over the extracted subgraph.
    """

    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')




class RWR_Subgraph_Sampler(nn.Module):
    """Module 3: Random Walk with Restart (RWR) for global subgraph extraction.
    
    Computes joint attention scores over historical edges and runs RWR
    to find the Top-N most relevant nodes.
    """
    def __init__(self, h_dim, top_n=200, alpha_restart=0.2, k_iter=3):
        super().__init__()
        self.h_dim = h_dim
        self.top_n = top_n
        self.alpha_restart = alpha_restart
        self.k_iter = k_iter
        
        self.W_time = nn.Linear(1, h_dim)
        self.W_a = nn.Linear(h_dim * 3, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, g, h_r_0, e_input):
        import dgl.nn as dglnn
        device = h_r_0.device
        with g.local_scope():
            # 1. Compute alpha_{u, v, r', \tau}
            def compute_alpha(edges):
                rel_feat = h_r_0[edges.data['type']]
                q_rel_feat = e_input[edges.src['id'].squeeze(-1)]
                # Map frequency back to continuous time encoding as a proxy for \Delta t_{\tau}
                t_enc = self.W_time(edges.data['fre'].float().unsqueeze(-1))
                cat_feat = torch.cat([rel_feat, q_rel_feat, t_enc], dim=-1)
                alpha = self.leaky_relu(self.W_a(cat_feat))
                return {'alpha': alpha}
            
            g.apply_edges(compute_alpha)
            
            # 2. Compute transition probability P_{u, v} via out-edge softmax
            rev_g = dgl.reverse(g, copy_edata=True)
            P = dglnn.edge_softmax(rev_g, rev_g.edata['alpha'])
            g.edata['P'] = P
            
            # 3. RWR Iteration
            e_s_mask = (e_input.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
            e_s_sum = e_s_mask.sum()
            e_s = e_s_mask / e_s_sum if e_s_sum > 0 else e_s_mask
            
            node_ids = g.ndata['id'].squeeze(-1)
            g_e_s = e_s[node_ids]
            
            g.ndata['pi'] = g_e_s.clone()
            
            def rwr_msg(edges):
                return {'m': edges.data['P'] * edges.src['pi']}
            
            def rwr_reduce(nodes):
                return {'pi_new': torch.sum(nodes.mailbox['m'], dim=1)}
                
            for _ in range(self.k_iter):
                g.update_all(rwr_msg, rwr_reduce)
                pi_new = g.ndata.get('pi_new', torch.zeros_like(g.ndata['pi']))
                g.ndata['pi'] = (1 - self.alpha_restart) * pi_new + self.alpha_restart * g_e_s
                
            # 4. Top-N Node Selection
            pi_scores = g.ndata['pi'].squeeze(-1)
            num_nodes_g = g.number_of_nodes()
            top_n_k = min(self.top_n, num_nodes_g)
            _, top_n_indices = torch.topk(pi_scores, top_n_k)
            
            # 5. Induce Subgraph
            top_n_subg = dgl.node_subgraph(g, top_n_indices)
            top_n_pi = top_n_subg.ndata['pi']
            
            return top_n_subg, top_n_pi


class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, pre_weight=0.7, discount=0, angle=0,
                 use_llm_prior=True, llm_emb_dir=None, llm_text_dim=1536, pre_type='short',
                 use_cl=False, temperature=0.007, tau_hard=0.1, cl_weight=0.5,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu=0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.pre_weight = pre_weight
        self.discount = discount
        self.use_llm_prior = use_llm_prior
        self.llm_text_dim = llm_text_dim
        self.pre_type = pre_type
        self.use_cl = use_cl
        self.temp = temperature
        # Module 4: temperature for the adaptive hard-negative weight
        #     beta_k = exp(sim(Z_local(q), Z_global(q_k^-)) / tau_hard)
        # and the cross-entropy / contrastive trade-off coefficient lambda.
        self.tau_hard = tau_hard
        self.cl_weight = cl_weight
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        # Module 2: query joint feature q_emb = MLP([h_s^(0) || h_r^(0)]).
        # The original LogCL used a single linear layer (self.w1); the paper's
        # Module 2 specifies a multi-layer perceptron, so we upgrade it to
        # two linear layers with a non-linearity in between.
        self.local_query_mlp = nn.Sequential(
            nn.Linear(self.h_dim * 2, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.w3 = nn.Linear(self.h_dim, self.h_dim)
        self.w4 = nn.Linear(self.h_dim*2, self.h_dim)
        self.w5 = nn.Linear(self.h_dim, self.h_dim)
        self.w6 = nn.Linear(self.h_dim,self.h_dim)
        self.w7 = nn.Linear(self.h_dim, self.h_dim)

        # Module 4: feature fusion MLP for the prediction stage.
        #     Z_fuse(s) = W_out * ReLU(W_in * [Z_local(s) || Z_global(s)])
        # The two-layer MLP replaces the original ConvTransE weighted-sum
        # mixing of Z_local and Z_global controlled by ``pre_weight``.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.h_dim * 2, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        # Sinusoidal time encoding parameters reused for Delta t_tau in Module 2's
        # query-aware time attention (Delta t_enc = cos(weight_t2 * Delta t + bias_t2)).
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))

        # Module 2: query-aware time attention parameters.
        #     e_tau = v^T tanh(W_1 h_{i,tau} + W_2 q_emb + W_3 Delta t_tau)
        self.local_W1 = nn.Linear(self.h_dim, self.h_dim)
        self.local_W2 = nn.Linear(self.h_dim, self.h_dim)
        self.local_W3 = nn.Linear(self.h_dim, self.h_dim)
        self.local_v = nn.Parameter(torch.empty(self.h_dim))
        nn.init.normal_(self.local_v, mean=0.0, std=0.02)

        self.weight_1 = nn.Linear(self.h_dim*2, self.h_dim)
        self.weight_2 = nn.Linear(self.h_dim*2, self.h_dim)
        self.bias = nn.Parameter(torch.zeros(1))

        self.weight_3 = nn.Linear(self.h_dim, 1)
        self.weight_4 = nn.Linear(self.h_dim, 1)
        self.bias_r = nn.Parameter(torch.zeros(1))

        # h_e^struct, h_r^struct: random structural embeddings (will be summed with the
        # LLM-projected text view to form the 0-th layer features h_e^(0), h_r^(0)).
        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        # ---------- Module 1: LLM-driven semantic prior injection ----------
        # Loads the offline pre-computed text-view embeddings produced by
        # data/encode_llm_prior.py and learns a linear projection W_proj that
        # maps them into the structural h_dim space. The projection is followed
        # by LayerNorm and added residually to the structural embeddings:
        #     h_e^(0) = h_e^struct + LayerNorm(W_proj_e @ h_e^text)
        #     h_r^(0) = h_r^struct + LayerNorm(W_proj_r @ h_r^text)
        # The LLM itself stays frozen and is never invoked at training time.
        if self.use_llm_prior:
            if llm_emb_dir is None:
                raise ValueError("use_llm_prior=True but llm_emb_dir is not provided")
            ent_path = os.path.join(llm_emb_dir, "entity_text_emb.pt")
            rel_path = os.path.join(llm_emb_dir, "relation_text_emb.pt")
            if not (os.path.isfile(ent_path) and os.path.isfile(rel_path)):
                raise FileNotFoundError(
                    "LLM text embeddings not found in {}. Run "
                    "`python data/encode_llm_prior.py -d <DATASET>` first.".format(llm_emb_dir))
            ent_text = torch.load(ent_path, map_location="cpu").float()
            rel_text = torch.load(rel_path, map_location="cpu").float()
            assert ent_text.shape == (num_ents, llm_text_dim), \
                f"entity_text_emb shape {tuple(ent_text.shape)} != ({num_ents},{llm_text_dim})"
            assert rel_text.shape == (num_rels, llm_text_dim), \
                f"relation_text_emb shape {tuple(rel_text.shape)} != ({num_rels},{llm_text_dim})"
            # Mirror the relation text view for the inverse relations so it aligns
            # with the [2*num_rels, h_dim] layout of self.emb_rel.
            rel_text_full = torch.cat([rel_text, rel_text], dim=0)
            self.register_buffer("entity_text_emb", ent_text, persistent=False)
            self.register_buffer("relation_text_emb", rel_text_full, persistent=False)
            self.W_proj_ent = nn.Linear(llm_text_dim, self.h_dim)
            self.W_proj_rel = nn.Linear(llm_text_dim, self.h_dim)
            self.ln_ent_prior = nn.LayerNorm(self.h_dim)
            self.ln_rel_prior = nn.LayerNorm(self.h_dim)
            
            # Gating parameters for Module 1 adaptive fusion
            self.W_g_ent = nn.Linear(self.h_dim * 2, self.h_dim)
            self.W_g_rel = nn.Linear(self.h_dim * 2, self.h_dim)

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        # Module 3: RWR Sampler for global subgraph
        self.rwr_sampler = RWR_Subgraph_Sampler(h_dim, top_n=200, alpha_restart=0.2, k_iter=3)

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.his_rgcn_layer = RGCNCell2(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)
        
        self.rgat_layer = RGAT(self.h_dim, self.h_dim, activation=F.rrelu, dropout=dropout, self_loop=True)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)   

        self.pre_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.pre_gate_weight, gain=nn.init.calculate_gain('relu'))

        # Module 2 replaced the GRU-based local sequential aggregator with a
        # query-aware time attention; the previous self.entity_cell / self.relation_cell
        # GRU cells are therefore no longer instantiated.

        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.decoder_ob1 = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError 

    def _fuse_prior(self):
        """Compute h_e^(0) and h_r^(0) by injecting the LLM text prior
        into the structural embeddings via a learnable linear projection
        and an adaptive gating mechanism.
        Returns:
            h_e_0: [num_ents, h_dim]
            h_r_0: [2 * num_rels, h_dim]
        """
        if self.use_llm_prior:
            ent_text_proj = self.ln_ent_prior(self.W_proj_ent(self.entity_text_emb))
            g_ent = torch.sigmoid(self.W_g_ent(torch.cat([self.dynamic_emb, ent_text_proj], dim=-1)))
            h_e_0 = g_ent * self.dynamic_emb + (1 - g_ent) * ent_text_proj
            
            rel_text_proj = self.ln_rel_prior(self.W_proj_rel(self.relation_text_emb))
            g_rel = torch.sigmoid(self.W_g_rel(torch.cat([self.emb_rel, rel_text_proj], dim=-1)))
            h_r_0 = g_rel * self.emb_rel + (1 - g_rel) * rel_text_proj
        else:
            h_e_0 = self.dynamic_emb
            h_r_0 = self.emb_rel
        return h_e_0, h_r_0

    def forward(self, sub_graph, T_idx, query_mask, g_list, static_graph, use_cuda, e_input=None):
        # static_graph is kept for backward compatibility with the call sites and
        # is not used after Module 1 replaced the e-w-graph branch with the
        # LLM-driven semantic prior injection.
        del static_graph

        # Module 1: build h_e^(0) and h_r^(0) from the LLM semantic prior.
        h_e_0, h_r_0 = self._fuse_prior()
        self.h = F.normalize(h_e_0) if self.layer_norm else h_e_0
        static_emb = None  # legacy slot kept in the return tuple

        #-----------------全局历史建模 (Module 3) -------------------------------------
        # e_input carries the per-entity pooled query relation; for query subjects this
        # is the average of their connected query relation embeddings, zero elsewhere.
        # It is forwarded into the global RWR subgraph as h_r^l for the relation
        # modulation in UnionRGCNLayer3.
        self.his_ent, subg_index = self.all_GCN(self.h, sub_graph, use_cuda,
                                                rel_emb=h_r_0,
                                                q_rel_per_node=e_input)
        his_r_emb = F.normalize(h_r_0)
        his_att = F.softmax(self.w5(query_mask + self.his_ent), dim=1)
        his_emb = his_att * self.his_ent
        his_emb = F.normalize(his_emb)

        # ----------------- Module 2: Query-Guided Local Entity Encoder -----------------
        # Spatial view: per-snapshot R-GCN whose 0-th layer features are reset to the
        # LLM-initialised h^(0) (no cross-snapshot recurrent state).
        # Temporal view: query-aware attention over the m snapshot summaries.
        his_temp_embs = []        # h_{i,tau}^(L)  per snapshot
        his_rel_embs = []         # evolved relation reps per snapshot (input for Module 4)
        snapshot_summaries = []   # H_tau = AvgPool over local subgraph nodes
        delta_t_encs = []         # Delta t_tau = cos(weight_t2 * Delta t + bias_t2)

        if self.pre_type == "all":
            m = len(g_list)
            base_ent = self.h           # h^(0): shared 0-th layer node features
            base_rel = h_r_0            # h_r^(0): shared 0-th layer relation features
            for i, g in enumerate(g_list):
                g = g.to(self.gpu)

                temp_e = base_ent[g.r_to_e]
                x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() \
                    if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1], :]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean
                x_input = base_rel + x_input

                # h_{i,tau}^(L): L-layer R-GCN starting from h^(0) on the local snapshot.
                current_h = self.rgcn.forward(g, base_ent, [base_rel, base_rel])
                current_h = F.normalize(current_h) if self.layer_norm else current_h

                # H_tau = AvgPool({h_{i,tau}^(L) | i in V(G_tau^local)})
                H_tau = current_h.mean(dim=0, keepdim=True)            # [1, h_dim]

                # Delta t_tau: g_list[0] is the oldest, so Delta t = m - i.
                delta_t = float(m - i)
                dt_enc = torch.cos(self.weight_t2 * delta_t + self.bias_t2)  # [1, h_dim]

                # Relation evolution per snapshot, used by Module 4 contrastive learning.
                time_weight = F.sigmoid(torch.mm(x_input, self.time_gate_weight) + self.time_gate_bias)
                self.hr = time_weight * x_input + (1 - time_weight) * base_rel
                self.hr = F.normalize(self.hr) if self.layer_norm else self.hr

                his_temp_embs.append(current_h)
                his_rel_embs.append(self.hr)
                snapshot_summaries.append(H_tau)
                delta_t_encs.append(dt_enc)

            # Stack the per-snapshot tensors for a vectorised attention computation.
            h_e_tau_stack = torch.stack(his_temp_embs, dim=0)            # [m, num_ents, h_dim]
            H_stack = torch.cat(snapshot_summaries, dim=0)               # [m, h_dim]
            dt_stack = torch.cat(delta_t_encs, dim=0)                    # [m, h_dim]

            # q_emb is supplied externally via query_mask, which is non-zero only for
            # query subjects (the local_query_mlp output) and zero elsewhere.
            q_emb = query_mask.unsqueeze(0).expand_as(h_e_tau_stack)     # [m, num_ents, h_dim]
            dt_expanded = dt_stack.unsqueeze(1).expand(m, self.num_ents, self.h_dim)

            # e_tau = v^T tanh(W_1 h_{i,tau} + W_2 q_emb + W_3 Delta t_tau)
            attn_inner = torch.tanh(
                self.local_W1(h_e_tau_stack)
                + self.local_W2(q_emb)
                + self.local_W3(dt_expanded)
            )                                                            # [m, num_ents, h_dim]
            e_tau = torch.matmul(attn_inner, self.local_v)               # [m, num_ents]

            # alpha_tau = softmax over the m historical snapshots.
            alpha = F.softmax(e_tau, dim=0)                              # [m, num_ents]

            # Z_local[i] = sum_tau alpha_tau(i) * H_tau
            history_emb = alpha.transpose(0, 1) @ H_stack                # [num_ents, h_dim]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            self.hr = None
            history_emb = None

        return history_emb, static_emb, self.hr, his_emb, his_r_emb, his_temp_embs, his_rel_embs, subg_index


    def predict(self, que_pair, sub_graph, T_id, test_graph, num_rels, static_graph, test_triplets, use_cuda):
        with torch.no_grad():
            all_triples = test_triplets

            # Inject the LLM semantic prior so that the query construction below
            # operates on h_e^(0) / h_r^(0) instead of the raw structural embeddings.
            h_e_0, h_r_0 = self._fuse_prior()

            #-----------------查询数据处理-------------------------------------
            uniq_e = que_pair[0]
            r_len = que_pair[1]
            r_idx = que_pair[2]
            temp_r = h_r_0[r_idx]
            e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
            for span, e_idx in zip(r_len, uniq_e):
                x = temp_r[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                e_input[e_idx] = x_mean

            query_mask = torch.zeros((self.num_ents, self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
            e1_emb = h_e_0[uniq_e]
            rel_emb = e_input[uniq_e]   # 实体所连的所有关系池化
            # Module 2: q_emb = MLP([h_s^(0) || h_r^(0)])
            query_emb = self.local_query_mlp(torch.concat([e1_emb, rel_emb], dim=1))
            query_mask[uniq_e] = query_emb

            embedding, _, r_emb, his_emb, his_r_emb, _, _, _ = self.forward(sub_graph, T_id, query_mask, test_graph, static_graph, use_cuda, e_input=e_input)

            if self.pre_type == "all":
                # Module 4: feature fusion
                #     Z_fuse = W_out * ReLU(W_in * [Z_local || Z_global])
                # Score(s, r, o, t_q) = Decoder(Z_fuse(s), h_r, h_o^(0)).
                z_fused = self.fusion_mlp(torch.cat([embedding, his_emb], dim=1))
                scores_ob = self.decoder_ob.forward(z_fused, h_e_0, r_emb, all_triples)
                scores_en = F.log_softmax(scores_ob, dim=1)
            else:
                # The reduced "long"/"short" decoding paths from the original
                # LogCL pipeline are not part of the new four-module framework.
                raise NotImplementedError(
                    "pre_type='{}' is no longer supported by the four-module "
                    "pipeline; please pass --pre-type all.".format(self.pre_type))
            return all_triples, scores_en


    def get_loss(self,que_pair, sub_graph,T_idx, glist, triples, static_graph, use_cuda):
        """
        :param glist:
        :param triplets:
        :param static_graph: 
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_cl = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        
        all_triples = triples

        # Inject the LLM semantic prior so that the query construction below
        # operates on h_e^(0) / h_r^(0) instead of the raw structural embeddings.
        h_e_0, h_r_0 = self._fuse_prior()

        ### --------------查询数据处理-----------------------
        uniq_e = que_pair[0]
        r_len = que_pair[1]
        r_idx = que_pair[2]
        temp_r = h_r_0[r_idx]
        e_input = torch.zeros(self.num_ents, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_ents, self.h_dim).float()
        for span, e_idx in zip(r_len, uniq_e):
            x = temp_r[span[0]:span[1], :]
            x_mean = torch.mean(x, dim=0, keepdim=True)
            e_input[e_idx] = x_mean

        query_mask = torch.zeros((self.num_ents, self.h_dim)).to(self.gpu) if use_cuda else torch.zeros(1)
        # Module 2: q_emb = MLP([h_s^(0) || h_r^(0)])
        e1_emb = h_e_0[uniq_e]
        rel_emb = e_input[uniq_e]
        query_emb = self.local_query_mlp(torch.concat([e1_emb, rel_emb], dim=1))
        query_mask[uniq_e] = query_emb

        embedding, static_emb, r_emb, his_emb, his_r_emb, his_temp_embs, his_rel_embs, subg_index = self.forward(sub_graph, T_idx, query_mask, glist, static_graph, use_cuda, e_input=e_input)

        if self.pre_type == "all":
            # Module 4: feature fusion
            #     Z_fuse(s) = W_out * ReLU(W_in * [Z_local(s) || Z_global(s)])
            # The KG decoder then scores every candidate o using h_o^(0):
            #     Score(s, r, o, t_q) = Decoder(Z_fuse(s), h_r, h_o^(0)).
            z_fused = self.fusion_mlp(torch.cat([embedding, his_emb], dim=1))
            scores_ob = self.decoder_ob.forward(z_fused, h_e_0, r_emb, all_triples)
            loss_ent += F.cross_entropy(scores_ob, triples[:, 2], label_smoothing=0.1)

            if self.relation_prediction:
                score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
                loss_rel += self.loss_r(score_rel, all_triples[:, 1])

            # Module 4: hard-negative-aware InfoNCE between Z_local(q) and Z_global(q),
            # restricted to the structural hard negatives provided by the RWR sub-graph.
            if self.use_cl:
                # Semantic hard negative construction
                s_idx = triples[:, 0]
                r_idx = triples[:, 1]
                s_text_proj = self.ln_ent_prior(self.W_proj_ent(self.entity_text_emb[s_idx]))
                r_text_proj = self.ln_rel_prior(self.W_proj_rel(self.relation_text_emb[r_idx]))
                semantic_neg = s_text_proj + r_text_proj

                loss_cl = loss_cl + self.get_loss_cl(embedding, his_emb, z_fused, h_e_0, semantic_neg, triples, subg_index)
        else:
            raise NotImplementedError(
                "pre_type='{}' is no longer supported by the four-module "
                "pipeline; please pass --pre-type all.".format(self.pre_type))

        return loss_ent, loss_rel, loss_static, loss_cl

    def all_GCN(self, ent_emb, sub_graph, use_cuda, rel_emb=None, q_rel_per_node=None):
        # rel_emb defaults to self.emb_rel (raw structural relation embeddings) so that
        # external callers can inject the LLM-fused h_r^(0) instead.
        if rel_emb is None:
            rel_emb = self.emb_rel
        sub_graph = sub_graph.to(self.gpu)
        
        if q_rel_per_node is None:
            q_rel_per_node = torch.zeros_like(ent_emb)
            
        # Module 3: Global subgraph RWR sampling
        top_n_subg, top_n_pi = self.rwr_sampler(sub_graph, rel_emb, q_rel_per_node)
        
        # Extracted global subgraph node indices
        node_ids = top_n_subg.ndata['id'].squeeze(-1)
        
        # R-GCN over the extracted global subgraph
        # We do NOT multiply the features by pi, as pi is only used to select the Top-N nodes.
        his_emb_top_n = self.his_rgcn_layer.forward(top_n_subg, ent_emb, [rel_emb, rel_emb])
        
        # Scatter back to full entity space
        his_emb = torch.zeros_like(ent_emb)
        his_emb[node_ids] = his_emb_top_n
        
        subg_index = node_ids
        return F.normalize(his_emb), subg_index
    
    def get_loss_cl(self, z_local_full, z_global_full, z_fuse_full, h_e_0, semantic_neg, triples, subg_index):
        """Module 4: Hard-negative-aware local-global contrastive loss with 3 decoupled views."""
        device = z_local_full.device
        s_idx = triples[:, 0]
        o_idx = triples[:, 2]

        # Extract anchor embeddings for the query
        Z_local = F.normalize(z_local_full[s_idx], dim=1)
        Z_global = F.normalize(z_global_full[s_idx], dim=1)
        Z_fuse = F.normalize(z_fuse_full[s_idx], dim=1)

        # Positive and candidate embeddings
        h_pos = F.normalize(h_e_0[o_idx], dim=1)
        h_all = F.normalize(h_e_0, dim=1)
        
        # Semantic negative
        h_sem = F.normalize(semantic_neg, dim=1)
        
        # Similarities with positive
        sim_pos_fuse = (Z_fuse * h_pos).sum(dim=1) / self.temp
        sim_pos_local = (Z_local * h_pos).sum(dim=1) / self.temp
        sim_pos_global = (Z_global * h_pos).sum(dim=1) / self.temp
        
        # Similarities with structural negatives
        sim_all_fuse = Z_fuse @ h_all.T
        sim_all_local = Z_local @ h_all.T
        sim_all_global = Z_global @ h_all.T
        
        # Similarities with semantic negative
        sim_sem_fuse = (Z_fuse * h_sem).sum(dim=1, keepdim=True)
        sim_sem_local = (Z_local * h_sem).sum(dim=1, keepdim=True)
        sim_sem_global = (Z_global * h_sem).sum(dim=1, keepdim=True)
        
        # Structural hard negative mask (using top-N from RWR)
        hard_mask = torch.zeros(Z_fuse.size(0), self.num_ents, device=device)
        hard_mask[:, subg_index] = 1.0
        hard_mask.scatter_(1, o_idx.unsqueeze(1), 0.0) # exclude true tail
        
        # Shared beta (adaptive weight log representation) computed from Z_fuse
        beta_log_all = sim_all_fuse / self.tau_hard
        beta_log_sem = sim_sem_fuse / self.tau_hard
        
        # --- View 1: Fusion Contrast ---
        log_struct_neg_fuse = beta_log_all + (sim_all_fuse / self.temp)
        log_struct_neg_fuse = log_struct_neg_fuse.masked_fill(hard_mask == 0, float('-inf'))
        log_sem_neg_fuse = beta_log_sem + (sim_sem_fuse / self.temp)
        combined_fuse = torch.cat([sim_pos_fuse.unsqueeze(1), log_struct_neg_fuse, log_sem_neg_fuse], dim=1)
        log_denom_fuse = torch.logsumexp(combined_fuse, dim=1)
        L_fuse = (log_denom_fuse - sim_pos_fuse).mean()
        
        # --- View 2: Local Contrast ---
        log_struct_neg_local = beta_log_all + (sim_all_local / self.temp)
        log_struct_neg_local = log_struct_neg_local.masked_fill(hard_mask == 0, float('-inf'))
        log_sem_neg_local = beta_log_sem + (sim_sem_local / self.temp)
        log_special_neg_local = (Z_local * Z_global).sum(dim=1, keepdim=True) / self.temp
        combined_local = torch.cat([sim_pos_local.unsqueeze(1), log_struct_neg_local, log_sem_neg_local, log_special_neg_local], dim=1)
        log_denom_local = torch.logsumexp(combined_local, dim=1)
        L_local = (log_denom_local - sim_pos_local).mean()
        
        # --- View 3: Global Contrast ---
        log_struct_neg_global = beta_log_all + (sim_all_global / self.temp)
        log_struct_neg_global = log_struct_neg_global.masked_fill(hard_mask == 0, float('-inf'))
        log_sem_neg_global = beta_log_sem + (sim_sem_global / self.temp)
        log_special_neg_global = log_special_neg_local # symmetric similarity
        combined_global = torch.cat([sim_pos_global.unsqueeze(1), log_struct_neg_global, log_sem_neg_global, log_special_neg_global], dim=1)
        log_denom_global = torch.logsumexp(combined_global, dim=1)
        L_global = (log_denom_global - sim_pos_global).mean()
        
        return L_fuse + L_local + L_global