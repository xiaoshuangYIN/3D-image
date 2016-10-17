#include "graph-cut.h"
#include "opencv2/core/core.hpp"

#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/read_dimacs.hpp>

#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace std;

/*******************
 * Correspondences *
 *******************/

bool GraphCutDisparity::is_active(Correspondence c) {
  return (pair->disparity_left.at<uchar>(c.y, c.x) == -c.d);
}

/*****************
 * Min-Cut Graph *
 *****************/

long GraphCutDisparity::correspondence_hash(Correspondence c)
{
  long hash_key = c.x;

  hash_key *= pair->rows;
  hash_key += c.y;

  hash_key *= (max_disparity - min_disparity + 1);
  hash_key += (c.d - min_disparity);
  
  return hash_key;
}

GraphCutDisparity::node_index GraphCutDisparity::get_index(Correspondence c)
{
  long hash_key = correspondence_hash(c);
  return hash_to_graph_index[hash_key];
}

GraphCutDisparity::Vertex GraphCutDisparity::get_vertex(Correspondence c)
{
  node_index i = get_index(c);
  return boost::vertex(i, g);
}

void GraphCutDisparity::add_node(Correspondence c)
{
  // Add to graph
  Vertex node = boost::add_vertex(g);

  // Record the index
  long hash_key = correspondence_hash(c);
  node_index idx = boost::get(vertex_indices, node);
  hash_to_graph_index[hash_key] = idx;
}

void GraphCutDisparity::add_edge(Correspondence c1, Correspondence c2,
    edge_weight w_uv, edge_weight w_vu)
{
  Vertex u = get_vertex(c1);
  Vertex v = get_vertex(c2);

  Edge e, e_reverse;
  // Add edges
  tie(e, std::ignore) = boost::add_edge(u, v, g);
  tie(e_reverse, std::ignore) = boost::add_edge(v, u, g);
  // Give edges the appropriate weight
  put(capacities, e, w_uv);
  put(capacities, e_reverse, w_vu);

  // Initialize properties for max flow calculation
  put(residual_capacities, e, 0);
  put(residual_capacities, e_reverse, 0);
  put(reverse_edges, e, e_reverse);
  put(reverse_edges, e_reverse, e);
}


// See add_edge for details
void GraphCutDisparity::add_source_edge(Correspondence c, edge_weight w)
{
  Vertex v = get_vertex(c);

  Edge e, e_reverse;
  tie(e, std::ignore) = boost::add_edge(source, v, g);
  tie(e_reverse, std::ignore) = boost::add_edge(v, source, g);
  put(capacities, e, w);
  put(capacities, e_reverse, 0);
  put(residual_capacities, e, 0);
  put(residual_capacities, e_reverse, 0);
  put(reverse_edges, e, e_reverse);
  put(reverse_edges, e_reverse, e);
  return;
}

// See add_edge for details
void GraphCutDisparity::add_sink_edge(Correspondence c, edge_weight w)
{
  Vertex u = get_vertex(c);

  Edge e, e_reverse;
  tie(e, std::ignore) = boost::add_edge(u, sink, g);
  tie(e_reverse, std::ignore) = boost::add_edge(sink, u, g);
  put(capacities, e, w);
  put(capacities, e_reverse, 0);
  put(residual_capacities, e, 0);
  put(residual_capacities, e_reverse, 0);
  put(reverse_edges, e, e_reverse);
  put(reverse_edges, e_reverse, e);
  return;
}

/**************
 * Cost Model *
 **************/

void GraphCutDisparity::for_each_active(function<void(Correspondence)> fn, int alpha) {
  for (int y = 0; y < pair->rows; y++) {
    for (int x = 0; x < pair->cols; x++) {
      Correspondence c;
      c.d = -pair->disparity_left.at<uchar>(y, x);
      if (c.d != NULL_DISPARITY and c.d != alpha) {
        c.x = x;
        c.y = y;
        fn(c);
      }
    }
  }
}

void GraphCutDisparity::for_each_alpha(function<void(Correspondence)> fn, int alpha) {
  for (int y = 0; y < pair->rows; y++) {
    for (int x = 0; x < pair->cols; x++) {
      Correspondence c;
      c.x = x;
      c.y = y;
      c.d = alpha;
      if (is_valid(c, alpha)) fn(c);
    }
  }
}


bool GraphCutDisparity::within_bounds(Correspondence c) {
return (
    c.x >= 0 and
    (c.x + c.d) >= 0 and
    c.y >= 0 and
    c.x < pair->cols and
    (c.x + c.d) < pair->cols and
    c.y < pair->rows
  );
}

// within image boundary
// active or has disparity alpha
bool GraphCutDisparity::is_valid(Correspondence c, int alpha) {
  return (
    within_bounds(c) and
    (
      is_active(c) or (c.d == alpha)
    )
  );
}

inline int square(int x) {return x * x;}

// squared error
GraphCutDisparity::edge_weight GraphCutDisparity::data_cost(Correspondence c)
{

  Vec3f col1 = pair->left.at<Vec3f>(c.y, c.x);
  Vec3f col2 = pair->right.at<Vec3f>(c.y, c.x + c.d);

  return square(cv::norm(col1 - col2));
}

GraphCutDisparity::edge_weight GraphCutDisparity::occ_cost(Correspondence c) {
  int occ_count = 0;
  if (left_occlusion_count.at<uchar>(c.y, c.x) == 1)
    occ_count++;
  if (right_occlusion_count.at<uchar>(c.y, c.x + c.d) == 1)
    occ_count++;
  return Cp * occ_count;
} 

void GraphCutDisparity::record_occlusion_count(Correspondence c, int alpha)
{
  left_occlusion_count.at<uchar>(c.y, c.x)++;
  right_occlusion_count.at<uchar>(c.y, c.x + c.d)++;
}

void GraphCutDisparity::record_occlusion_counts(int alpha)
{
  for_each_active(
    [this, alpha](Correspondence c) { record_occlusion_count(c, alpha); }
    , alpha
  );
  for_each_alpha(
    [this, alpha](Correspondence c) { record_occlusion_count(c, alpha); }
    , alpha
  );
}

void GraphCutDisparity::add_alpha_nodes(int alpha)
{
  for_each_alpha(
    [this, alpha](Correspondence c) { add_alpha_node(c, alpha); }
    , alpha
  );
}

void GraphCutDisparity::add_alpha_node(Correspondence c, int alpha){
  edge_weight source_w = data_cost(c);
  edge_weight sink_w = occ_cost(c);

  add_node(c);
  add_source_edge(c, source_w);
  add_sink_edge(c, sink_w);

  return;
} 

void GraphCutDisparity::add_active_nodes(int alpha)
{
  for_each_active(
    [this, alpha](Correspondence c) { add_active_node(c, alpha); }
    , alpha
  );
}

void GraphCutDisparity::add_active_node(Correspondence c, int alpha){
  edge_weight source_w = occ_cost(c);
  edge_weight sink_w = data_cost(c) + smooth_cost(c);

  add_node(c);
  add_source_edge(c, source_w);
  add_sink_edge(c, sink_w);
  return;
} 

GraphCutDisparity::edge_weight GraphCutDisparity::smooth_cost(Correspondence c){
  vector<Correspondence> neighbors = get_inactive_neighbors(c,c.d);

  return V_smooth * neighbors.size();
} 

vector<GraphCutDisparity::Correspondence> GraphCutDisparity::get_inactive_neighbors(Correspondence c, int alpha){
  vector<Correspondence> neighbors;
  vector<int> offset = {0, 0, 1, -1};

  // x +- 1, y +- 1 neighbors, where is_valid
  for (size_t i=0; i<offset.size(); i++) {
    Correspondence c_tmp = {c.x + offset[i], c.y + offset[offset.size()-1-i], c.d}; 

    if(within_bounds(c_tmp) and !is_valid(c_tmp, alpha) ) {
      neighbors.push_back(c_tmp);
    }
  }

  return neighbors;
}

void GraphCutDisparity::add_all_conflict_edges(int alpha)
{
  for_each_active(
    [this, alpha](Correspondence c) { add_conflict_edges(c, alpha); }
    , alpha
  );
}

void GraphCutDisparity::add_conflict_edges(Correspondence c, int alpha){
  vector<Correspondence> conflicts = get_conflicts(c,alpha);

  for (Correspondence c_tmp : conflicts) {
    add_edge(c, c_tmp, INT_MAX, Cp);
  }

  return;
}

vector<GraphCutDisparity::Correspondence> GraphCutDisparity::get_conflicts(Correspondence c, int alpha){
  vector<Correspondence> conflicts;

  if(is_active(c) and c.d != alpha ) {
    // check shared pixel
    Correspondence c_alpha = {c.x, c.y, alpha};
    if (is_valid(c_alpha, alpha)) {
      conflicts.push_back(c_alpha);
    }

    // check shared mapped pixel
    Correspondence c_mapped = { c.x + c.d - alpha, c.y, alpha};
    if (is_valid(c_mapped, alpha)) {
      conflicts.push_back(c_mapped);
    }
  }

  return conflicts;
}

void GraphCutDisparity::add_all_neighbor_edges(int alpha)
{
  for_each_active(
    [this, alpha](Correspondence c) { add_neighbor_edges(c, alpha); }
    , alpha
  );
  for_each_alpha(
    [this, alpha](Correspondence c) { add_neighbor_edges(c, alpha); }
    , alpha
  );
}

void GraphCutDisparity::add_neighbor_edges(Correspondence c, int alpha){
  vector<Correspondence> neighbors = get_neighbors(c,alpha);

  for (Correspondence c_tmp : neighbors) {
    if (correspondence_hash(c) > correspondence_hash(c_tmp)) {
      add_edge(c, c_tmp, V_smooth, V_smooth);
    }
  }

  return;
} 

vector<GraphCutDisparity::Correspondence> GraphCutDisparity::get_neighbors(Correspondence c, int alpha){
  vector<Correspondence> neighbors;
  vector<int> offset = {0, 0, 1, -1};

  // x +- 1, y +- 1 neighbors, where is_valid
  for (size_t i=0; i<offset.size(); i++) {
    Correspondence c_tmp = {c.x + offset[i], c.y + offset[offset.size()-1-i], c.d}; 

    if(is_valid(c_tmp, alpha) ) {
      neighbors.push_back(c_tmp);
    }
  }

  return neighbors;
} 

/*************
 * Algorithm *
 *************/

bool GraphCutDisparity::run_iteration()
{
  bool improved = false;
  for (int alpha = min_disparity; alpha <= max_disparity; alpha++) {
    improved = run_alpha_expansion(-alpha) || improved;
    // assert(run_alpha_expansion(-alpha) == false);
    cv::imshow("WIP", 2 * pair->disparity_left);
    cv::waitKey(50);
  }
  return improved;
}

bool GraphCutDisparity::run_alpha_expansion(int alpha)
{
  initialize_graph();

  record_occlusion_counts(alpha);
  add_active_nodes(alpha);
  add_alpha_nodes(alpha);

  add_all_conflict_edges(alpha);

  add_all_neighbor_edges(alpha);

  // Compute min cut
  boykov_kolmogorov_max_flow(g, source, sink);

  return update_correspondences(alpha);
}

void GraphCutDisparity::initialize_graph()
{
  // Clear and reset properties
  g.clear();
  capacities = get(boost::edge_capacity_t(), g);
  residual_capacities = get(boost::edge_residual_capacity_t(), g);
  reverse_edges = get(boost::edge_reverse_t(), g);
  vertex_indices = get(boost::vertex_index_t(), g);
  colors = get(boost::vertex_color_t(), g);

  hash_to_graph_index.clear();
  left_occlusion_count.setTo(0);
  right_occlusion_count.setTo(0);

  // Add source / sink
  source = boost::add_vertex(g);
  sink = boost::add_vertex(g);

  return;
}

bool GraphCutDisparity::update_correspondences(int alpha)
{
  Color black = boost::color_traits<Color>::black();

  bool changed = false;
  for_each_active(
    [this, black, &changed](Correspondence c) {

      Vertex node = get_vertex(c);
      Color col = boost::get(colors, node);
      if (col == black) // still active
        return;
      changed = true;
      pair->disparity_left.at<uchar>(c.y, c.x) = NULL_DISPARITY;
      pair->disparity_right.at<uchar>(c.y, c.x + c.d) = NULL_DISPARITY;
      assert(!is_active(c));
    }
    , alpha
  );

  for_each_alpha(
    [this, alpha, black, &changed](Correspondence c) {

      bool was_active = is_active(c);
      Vertex node = get_vertex(c);
      Color col = boost::get(colors, node);
      bool now_active = (col != black);

      if (now_active != was_active) {
        changed = true;
        if (now_active) {
          pair->disparity_left.at<uchar>(c.y, c.x) = -c.d;
          pair->disparity_right.at<uchar>(c.y, c.x + c.d) = -c.d;
          assert(is_active(c));
        } else {
          pair->disparity_left.at<uchar>(c.y, c.x) = NULL_DISPARITY;
          pair->disparity_right.at<uchar>(c.y, c.x + c.d) = NULL_DISPARITY;
          assert(!is_active(c));
        }
      }

    }
    , alpha
  );
    
  return changed;
}

GraphCutDisparity& GraphCutDisparity::compute(StereoPair &_pair)
{
  pair = &_pair;

  pair->disparity_left = cv::Mat(pair->rows, pair->cols, CV_8UC1);
  pair->disparity_right = cv::Mat(pair->rows, pair->cols, CV_8UC1);

  pair->disparity_left.setTo(NULL_DISPARITY);
  pair->disparity_right.setTo(NULL_DISPARITY);

  min_disparity = (pair->min_disparity_left < pair->min_disparity_right) ? pair->min_disparity_left : pair->min_disparity_right;
  min_disparity = min_disparity - 2;
  min_disparity = (min_disparity < 1) ? 1 : min_disparity;
  max_disparity = (pair->max_disparity_left > pair->max_disparity_right) ? pair->max_disparity_left : pair->max_disparity_right;
  max_disparity = max_disparity + 2;
  max_disparity = (max_disparity > 255) ? 255 : max_disparity;

  left_occlusion_count = cv::Mat(pair->rows, pair->cols, CV_8UC1);
  right_occlusion_count = cv::Mat(pair->rows, pair->cols, CV_8UC1);

  cv::imshow("Key", 2 * pair->true_disparity_left);
  cv::waitKey(50);

  for (int i = 0; i < num_iters; i++) {
    run_iteration();
  }

  return *this;
}

GraphCutDisparity::GraphCutDisparity(int _Cp, int _V) {
  Cp = _Cp;
  V_smooth = _V;
  return;
}