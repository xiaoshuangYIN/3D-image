#pragma once
#include "disparity-algorithm.h"

#include <map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>

#include <climits>

#include <functional>

class GraphCutDisparity : public DisparityAlgorithm {
private:


  /*******************
   * Correspondences *
   *******************/

  StereoPair *pair;

  typedef struct _Correspondence {
    int x;
    int y;
    int d; // disparity
  } Correspondence;

  int min_disparity;
  int max_disparity;

  /**
   * Use the disparity map in the pair variable
   * to determine if the disparity at this pixel
   * is the same as the Correspondence
   */
  bool is_active(Correspondence c);

  /*****************
   * Min-Cut Graph *
   *****************/

  typedef int node_index;
  typedef int edge_weight;


  int NULL_DISPARITY = 0;


  /* Boost Graph Library code for use with min-cut algorithm */
  typedef boost::vecS OutEdgeListT;
  typedef boost::vecS VertexListT;
  typedef boost::directedS DirectedT;

  typedef boost::adjacency_list_traits <OutEdgeListT, VertexListT, DirectedT> Traits;

  typedef Traits::edge_descriptor Edge;
  typedef Traits::vertex_descriptor Vertex;

  typedef boost::default_color_type Color;

  typedef
      boost::property < boost::vertex_index_t, node_index,
      boost::property < boost::vertex_color_t, Color,
      boost::property < boost::vertex_distance_t, edge_weight,
      boost::property < boost::vertex_predecessor_t, Edge> > > >
    VertexPropertiesT;

  typedef
      boost::property < boost::edge_capacity_t, edge_weight,
      boost::property < boost::edge_residual_capacity_t, edge_weight,
      boost::property < boost::edge_reverse_t, Edge> > >
    EdgePropertiesT;

  typedef boost::adjacency_list <
      OutEdgeListT, VertexListT, DirectedT,
      VertexPropertiesT, EdgePropertiesT >
    Graph;

  typedef boost::property_map<Graph, boost::edge_capacity_t>::type CapacityMap;
  typedef boost::property_map<Graph, boost::edge_residual_capacity_t>::type ResidualMap;
  typedef boost::property_map<Graph, boost::edge_reverse_t>::type ReverseMap;
  typedef boost::property_map<Graph, boost::vertex_color_t>::type ColorMap;
  typedef boost::property_map<Graph, boost::vertex_index_t>::type VertexIndexMap;

  VertexIndexMap vertex_indices;
  CapacityMap capacities;
  ResidualMap residual_capacities;
  ReverseMap reverse_edges;
  ColorMap colors;

  /**
   * The min-cut graph itself
   */
  Graph g;

  Vertex source;
  Vertex sink;

  /**
   * Correspondences must be represented by nodes in the graph
   * that have sequential indices. We keep track of these indices in
   * a map */
  std::map<long, node_index> hash_to_graph_index;
  /** Use a hashing function to make index lookup easier */
  long correspondence_hash(Correspondence c);
  /** Get index of the node in the graph representing c */
  node_index get_index(Correspondence c);
  /** Get the node itself that represents correspondence c */
  Vertex get_vertex(Correspondence c);


  /** Add a node to the graph representing c */
  void add_node(Correspondence c);

  /** Add neighbor constraints to correspondences c1 and c2 */
  void add_edge(Correspondence c1, Correspondence c2, edge_weight w_uv, edge_weight w_vu);

  /** Add edges that represent the unary costs
   * associated with a correspondence */
  void add_source_edge(Correspondence c, edge_weight w);
  void add_sink_edge(Correspondence c, edge_weight w);

  /**************
   * Cost Model *
   **************/

  edge_weight Cp;
  edge_weight V_smooth;

  /*
   * During each alpha expansion, we consider only the set of correspondences
   * that are currently active or have disparity alpha.
   */

  /** Active correspondences */
  void for_each_active(std::function<void(Correspondence)> fn, int alpha);
  /** Correspondences with disparity alpha */
  void for_each_alpha(std::function<void(Correspondence)> fn, int alpha);

  /**
   * Neither the left image pixel nor
   * right image pixel are out of bounds 
   */
  bool within_bounds(Correspondence c);

  /**
   * Correspondence is within the bounds
   * and is either active or has disparity alpha
   */
  bool is_valid(Correspondence c, int alpha);

  /** Cost of the match between two pixels, using squared error */
  edge_weight data_cost(Correspondence c);

  /** Occlusion cost if this correspondence is deactivated */
  edge_weight occ_cost(Correspondence c);
  /**
   * Count the number of possible correspondences being considered
   * that involve each pixel. If two correspondences involve the
   * pixel, the pixel cannot be occluded because one correspondence
   * must remain active.
   */
  cv::Mat left_occlusion_count, right_occlusion_count;
  void record_occlusion_count(Correspondence c, int alpha);
  void record_occlusion_counts(int alpha);

  /**
   * Add all correspondences with disparity alpha to the min-cut graph */
  void add_alpha_nodes(int alpha);
  /** Add a node to the graph representing a correspondence with disparity
   * alpha and add source/sink edges to represent model costs */
  void add_alpha_node(Correspondence c, int alpha);

  /**
   * Add all active correspondences to the min-cut graph */
  void add_active_nodes(int alpha);
  /**
   * Add a node to the graph representing an active correspondence
   * and add source/sink edges to represent model costs */
  void add_active_node(Correspondence c, int alpha);
  /** Smoothness cost w.r.t. inactive correspondences not considered
   * in the alpha expansion. */
  edge_weight smooth_cost(Correspondence c);
  std::vector<Correspondence> get_inactive_neighbors(Correspondence c, int alpha);

  /**
   * Use neighbor costs to enforce the constraint that every pixel is involved
   * in exactly one correspondence */
  void add_all_conflict_edges(int alpha);
  void add_conflict_edges(Correspondence c, int alpha);
  /**
   * Get all correspondences in the alpha expansion that could possibly conflict
   * with Correspondence c */
  std::vector<Correspondence> get_conflicts(Correspondence c, int alpha);

  /**
   * Use neighbor costs to enforce a smoothness constraint */
  void add_all_neighbor_edges(int alpha);
  void add_neighbor_edges(Correspondence c, int alpha);
  /**
   * Find all neighboring correspondences, i.e. correspondences in a
   * 4-connected region that have the same disparity */
  std::vector<Correspondence> get_neighbors(Correspondence c, int alpha);

  /*************
   * Algorithm *
   *************/

  int num_iters = 2; // Shows good results even if we stop early

  /**
   * Perform one iteration of the overall alpha expansion algorithm,
   * meaning run an alpha expansion for all possible values of alpha
   */
  bool run_iteration();

  /**
   * Perform an alpha expansion by setting up the graph, performing a min-cut,
   * and updating the correspondences
   *
   * Returns true if an update was performed, false if nothing changed
   */
  bool run_alpha_expansion(int alpha);

  /**
   * Clear the graph - a new min-cut graph must be generated
   * for every run. */
  void initialize_graph();

  /**
   * Use the results of the min-cut to update which correspondences are active.
   *
   * Returns true if an update was performed, false if nothing changed
   */
  bool update_correspondences(int alpha);

public:
  /**
   * Set up variables and run the graph cut algorithm */
  GraphCutDisparity& compute(StereoPair &pair);
  GraphCutDisparity(int _Cp, int _V);
};
