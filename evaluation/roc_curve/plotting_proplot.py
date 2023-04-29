import proplot as pplt
import numpy as np

from .roc_curve import compute_roc_curve


class ROC_Curve_proplot:
  """Plot ROC Curves using matplotlib."""

  def __init__(self, axis_label_size='22', axis_tick_size='22', legend_text_size='15'):

    """
    Parameters
    -----------

    axis_label_size: integer, default=22
        Controls font size of axis labels
    axis_tick_size: integer, default=14
        Controls tick marker size
    legend_text_size: integer, default=18
        Controls font size of legend text

    """

    self.axis_label_size = axis_label_size
    self.axis_tick_size = axis_tick_size
    self.legend_text_size = legend_text_size

  def plot(self, true_labels, predictions, folds=None, xrange=(0, 1), yrange=(0, 1),
           line_width='2', line_color=None, line_dash='-', legend_label=None,
           legend_location=None, ci_method='parametric', alpha=0.05, bootstrap_iters=0,
           direction='TPRvsFPR', x_scale='linear', thresholds=None, random_seed=0, save_dir=None):

    """Plot ROC curves using matplotlib

    Parameters
    -----------
    true_labels: pd.Dataframe
        A pandas dataframe containing the true labels.
    predictions: pd.Dataframe
        A pandas dataframe containing the classifier predictions.
    folds: pd.Dataframe, default=None
        A pandas dataframe containing the fold numbers used for cross-validation.
    xrange: tuple (min, max), default=(0,1)
        Controls x-range of plot.
    yrange: tuple (min, max), default=(0,1)
        Controls y-range of plot.
    line_width: integer, default=4
        Controls width of ROC curve line.
        Options include: 'linear' or 'log' scale for x-axis.
    line_color: string
        Bokeh color palettes can be found here:
        https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    line_dash: string, default='solid'
        Options include: 'solid', 'dashed', 'dashdot', 'dotdash'.
    legend_label: str, default=None
        Legend captions for plotted curves.
    legend_location: string, default=None
        Options include: None, 'top_right', 'bottomm_right', 'top_left', 'bottom_left'.
        None option does not show the legend.
    thresholds: pd.Dataframe or dict, default=None
        A list containing false positive rates over which to interpolate true positive.
        rate values.
    ci_method: string, default='parametric'
        Options include:'parametric' or 'bootstrap'
        The parametric method computes the standard error based on the number of folds and
        the user-defined significance level (alpha) for a t-distribution.
        The bootstrap method uses empirical quantiles from bootstrap samples generated
        by resampling the same number of sample scores and outcome values as the original
        data with replacement.
    alpha: integer, default=0.05
        Confidence level at which to compute the confidence interval
    bootstrap_iters: integer, default=100
        The number of bootstrap iterations used to compute the ROC curve and confidence interval.
    direction: string, default='TPRvsFPR'
        Options include: 'TPRvsFPR' and 'TNRvsFNR'
    x_scale: string, default='linear'
    random_seed: integer, default=0
        Controls the randomness and reproducibility of the indices for generating bootstrap samples.
    save_dir: str
        Path to directory at which to save the plot.

    Returns
    --------
    ROC curve(s) plotted with bokeh.

    """

    self.direction = direction
    self.x_scale = x_scale
    self.xrange = xrange
    self.yrange = yrange
    self.line_width = line_width
    self.line_dash = line_dash
    self.legend_location = legend_location

    if folds is None:
      folds = list(np.repeat(None, len(true_labels)))
    if line_color is None:
      line_color = list(Category10[10][:len(true_labels)])

    fig, ax = self._figure_initialize()

    for (truth, pred, fold, label, color) in zip(true_labels, predictions, folds,
                                                 legend_label, line_color):
      fpr_agg_roc_curve, fpr_ci, fnr_agg_roc_curve, fnr_ci = compute_roc_curve(truth, pred, fold,
                                                                               thresholds,
                                                                               ci_method,
                                                                               alpha,
                                                                               bootstrap_iters,
                                                                               random_seed)

      if self.direction == 'TPRvsFPR':
        agg_roc_curve, ci = fpr_agg_roc_curve, fpr_ci
      else:
        agg_roc_curve, ci = fnr_agg_roc_curve, fnr_ci

      self._plot_curve(ax, agg_roc_curve, ci, color, label)

    self._show_plot(fig, ax, save_dir)

  def _figure_initialize(self):

    """Initialize matplotlib figure."""
    
    pplt.rc['font.family'] = 'serif'
    
    fig = pplt.figure(figsize=(1.2*6, 7), facecolor='white')
    ax = fig.subplot(xlabel='Mean Predicted Probability', ylabel='Patient Count')

    if self.direction == 'TPRvsFPR':
      ax.set_ylabel('Sensitivity', fontsize=self.axis_label_size, color='black');
      if self.x_scale=='linear':
        ax.set_xlabel('1 - Specificity', fontsize=self.axis_label_size, color='black')
      elif self.x_scale=='log':
        ax.set_xlabel('log10(1 - Specificity)', fontsize=self.axis_label_size, color='black')

    if self.direction == 'TNRvsFNR':
      ax.set_ylabel('Specificity', fontsize=self.axis_label_size, color='black')
      if self.x_scale=='linear':
        ax.set_xlabel('1 - Sensitivity', fontsize=self.axis_label_size, color='black')
      elif self.x_scale=='log':
        ax.set_xlabel('log10(1 - Sensitivity)', fontsize=self.axis_label_size, color='black')

    rand_seq = np.linspace(0, 1, 10000)[1:-1]
    if self.legend_location is not None:
      ax.plot(rand_seq, rand_seq,color='black', linewidth=self.line_width, linestyle='--')
    else:
      ax.plot(rand_seq, rand_seq, color='black', linewidth=self.line_width, linestyle='--')

    ax.set_xscale(self.x_scale)
    ax.format(xlim=(self.xrange[0], self.xrange[1]))
    ax.format(ylim=(self.yrange[0], self.yrange[1]))
    ax.format(xlabelsize=self.axis_label_size)
    ax.format(ylabelsize=self.axis_label_size)
    ax.format(xticklabelsize=self.axis_tick_size)
    ax.format(yticklabelsize=self.axis_tick_size)
    ax.grid(True, ls=':', color='darkgray', linewidth=1.5)

    return fig, ax

  def _plot_curve(self, ax, agg_roc_curve, ci, color, label):
    """Plot ROC curve lines.

    Parameters
    -----------
    plot: matplotlib figure
    agg_roc_curve: pd.DataFrame
        A dataframe containing the aggregated true positive rate mean and standard
        deviation indexed by the user-defined false positive rate values used for
        interpolation.
    ci: pd.DataFrame
        A dataframe containing the lower and upper confidence interval bounds.
    line_color: string
        Line color
    legend_label: string
        Legend caption

    Returns
    --------
    matplotlib figure with plotted roc curve

    """

    if self.x_scale=='log':
      agg_roc_curve = agg_roc_curve.loc[lambda x: agg_roc_curve.index != 0]
      ci = ci.loc[lambda x: ci.index != 0]

    ax.fill_between(ci.index, ci['lower_bound'], ci['upper_bound'], alpha=0.1, color=color, label='')
    if self.legend_location is not None:
      ax.plot(agg_roc_curve.index, agg_roc_curve['tpr']['mean'], color=color,
              linewidth=self.line_width, linestyle=self.line_dash, label='%s' %(label))

    else:
      ax.plot(agg_roc_curve.index, agg_roc_curve['tpr']['mean'], color=color,
              linewidth=self.line_width, linestyle=self.line_dash)

  def _show_plot(self, fig, ax, save_dir=None):
    """Show ROC curve plot in a jupyter notebook and save file if save_dir is specified

    Parameters
    -----------
    plot: matplotlib figure with plotted roc curve(s)
    save_dir: string, default=None
        Directory where plot .png file should be saved

    Returns
    --------
    matplotlib figure with plotted roc curve in a jupyter notebook

    """

    if self.legend_location is not None:
      ax.legend(bbox_to_anchor =(0.5, -0.25), 
                loc=self.legend_location, 
                ncol=3, prop={'size': self.legend_text_size}
               )
    
    if save_dir is not None:
      fig.savefig(save_dir)
    else:
      fig.show()
