import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants
from scipy import integrate
from scipy import interpolate
import os


# Helper functions
def ft_i2k(im):
    """
    Fourier transform from image-space to k-space

    """
    return np.fft.fftshift(np.fft.ifftn(im))


def IsEven(number: int = None) -> bool:
    """
    Determine if the given integer is even.

    """
    return number % 2 == 0


def ft_k2i(ft):
    """
    Fourier transform from k-space to image-space

    """
    return np.fft.fftn(np.fft.fftshift(ft))

# Flow simulation classes
class __BaseSimulator:
    """
    Base class for simulator.

    This class provides common attributes and methods that can be used by all subclasses to perform simulations.

    """

    def __init__(self) -> None:
        # Sequence dependend variables
        self.preset_name = None
        self.partial_fourier_factor = None
        self.echospacing = None
        self.matrix_heigh = None
        self.matrix_width = None
        self.shots = None
        self.epi_factor = None
        self.snr = np.inf
        self.blip_fraction = None
        self.prephase_fraction_x = None
        self.prephase_fraction_y = None
        self.gradient_amplitude = None
        self.blip_amplitude = None
        self.flow_direction = None
        self.flow_velocity = None
        self.flow_acceleration = None
        self.echo_time_shifting = None
        self.starting_polartiy = None
        self.first_line_sampled = None
        self.setCounter = None
        self.prephaser = None
        self.gamma = physical_constants["proton gyromag. ratio in MHz/T"][0]
        self.offresonance = 0
        self.acceleration = 0
        self.trans_factor = 1e-2  # [mT*ms²/m] to [mT*ms²/cm]
        self.unsampled_gradient = False
        self.k_interleave = False
        self.idealizedProtocoll = False
        self.distributeMoments = True

        # Input image
        self.input_image = None

        # Computed variables
        self.timing_map = None
        self.ADC_dwelltime = None
        self.delta_ETS = None
        self.prephase_duration_x = None
        self.prephase_duration_y = None
        self.prephase_amplitude = None
        self.Time2FirstBlip = None
        self.moments_x = None
        self.moments_y = None
        self.gradient_matrix = None
        self.kSpace = None
        self.kSpace_phase = None
        self.PhaseFigure_x = None
        self.PhaseFigure_y = None

        # Sequence presets
        self.presets = {
            "miFB_paper": {
                "partial_fourier_factor": 1,
                "echospacing": 1.54,  # [ms]
                "matrix_heigh": 65,
                "matrix_width": 64,
                "shots": None,
                "epi_factor": 4,
                "snr": np.inf,
                "blip_fraction": 0.25,  # percentage of echospacing
                "gradient_amplitude": 34.5, # [mT/m]
                "blip_amplitude": 34,     # [mT/m]
                "offresonance": 0,
                "flow_direction": "FE",
                "flow_velocity": 1,  # [m/ms]
                "flow_acceleration": 0,  # [m/ms²]
                "echo_time_shifting": 1,
                "starting_polartiy": 0,
                "prephaser": True,
                "prephase_fraction_x": 0.5,
                "prephase_fraction_y": 0.3,
                "idealizedProtocoll": True,
            },
        }

        # Deafult figure parameters
        self.fig_para = {
            "color": "k",
            "rstride": 1,
            "cstride": 0,
            "linewidth": 0.8,
            "alpha": 0.8,
        }
        self.fig_para_c1 = "#1a5fb4ff"
        self.fig_para_c2 = "#e01b24ff"
        self.fig_para_c3 = "#26a269"
        self.fig_para_skip = 2
        self.figdir = None
        self.save_figures = True
        self.saveas = ".svg"

        self.ax_para = {
            "xlabel":"kx",
            "ylabel":"ky",
            "zlabel":"$M1_x$[mT*ms²/cm]",
            "xticks": [-1, 0, 1],
            "yticks": [-1, 0, 1],
            "zticks": [-2,-1, 0, 1, 2],     
            "xticklabels": ["$-k_{max}$",0,"$k_{max}$"],
            "yticklabels": ["$-k_{max}$",0,"$k_{max}$"],
        }

    def set_preset(self, preset_name):
        self.preset_name = preset_name

    def set_variable(self, key, value):
        """
        Set the specified variable in the instance.

        """
        # Check if the attribute exists
        if hasattr(self, key):
            setattr(self, key, value)
            self.presets[self.preset_name][key] = value
        else:
            raise AttributeError(f"Attribute '{key}' does not exist")

    def apply_preset(self):
        """
        Applies the specified preset values to the variables in the helper object.

        """
        if self.preset_name not in self.presets:
            raise AttributeError(f"Preset '{self.preset_name}' does not exist")

        # Apply all variables in the preset
        for key, value in self.presets[self.preset_name].items():
            self.set_variable(key, value)

        self.resolve_dependencies()

    def change_variable(self, key, value):
        """
        Changes the specified variable in the instance.

        """
        if key not in self.presets[self.preset_name]:
            raise AttributeError(f"Attribute '{key}' does not exist")
        self.presets[self.preset_name][key] = value

    def resolve_dependencies(self):
        """
        Resolves dependencies between matrix width, height, shots, and epi-factor.

        This method calculates missing values based on given parameters and ensures that
        matrix height is a multiple of either number of shots or Epi-factor. If not, it
        increases matrix width and height accordingly.

        The method also updates the epi_factor and shots attributes based on the calculated
        matrix width and other available parameters.

        Finally, it calculates the first line sampled based on the partial fourier factor 
        and some timing parameters.
        """
        # # Custom logic to calculate missing values
        if self.matrix_heigh is None and self.matrix_width is not None:
            self.matrix_heigh = self.matrix_width + 1  # Directly set the attribute

        # Check if Matrix height is multiple of either number of shots or Epi-factor
        # If not increase matrix_width and heigh
        # Check if Matrix height is a multiple of either number of shots or Epi-factor
        # If not, increase matrix_width and heigh
        if self.shots is not None:
            if self.matrix_width % self.shots != 0:
                self.matrix_width = self.shots * (self.matrix_heigh // self.shots + 1)
        elif self.epi_factor is not None:
            if self.matrix_width % self.epi_factor != 0:
                self.matrix_width = self.epi_factor * (
                    self.matrix_width // self.epi_factor + 1
                )

        if self.matrix_width is not None and self.shots is not None:
            self.epi_factor = self.matrix_width // self.shots

        if self.matrix_width is not None and self.epi_factor is not None:
            self.shots = self.matrix_width // self.epi_factor

        self.first_line_sampled = self.matrix_width - int(
            np.round(self.matrix_width * self.partial_fourier_factor)
        )
        self.SetSetCounter()

        # Compute duration and amplitude of prephaser
        if self.idealizedProtocoll and self.flow_direction != "PE":
            self.blip_duration = 0
        else:
            self.blip_duration = self.echospacing * self.blip_fraction
        
        # Compute timing parameters
        self.readout_duration = self.echospacing - self.blip_duration
        self.prephase_duration_x = self.prephase_fraction_x * self.readout_duration
        self.prephase_duration_y = self.prephase_fraction_y * self.readout_duration
        self.prephase_amplitude = self.gradient_amplitude / (
            2 * self.prephase_fraction_x
        )
        self.ADC_dwelltime = self.echospacing / self.matrix_heigh
        if self.starting_polartiy:
            self.prephase_amplitude *= -1

    def check_variables(self) -> None:
        """
        Checks if the value of a variable could lead to unwanted results.

        matrix height < 64: leads to phase jumps
        """
        if self.matrix_heigh <= 64:
            print(
                "Warning: Matrix height is less than 64. This could lead to unexpected results. Increasing matrix height to 64."
            )
            self.set_variable("matrix_heigh", 64)

        if self.flow_direction == "PE":
            self.blip_duration = self.echospacing * self.blip_fraction

    def ProduceTimeAndGradientVector(self, shot=0) -> tuple[np.array, np.array]:
        """
        Generates a timing and gradient map based on the given parameters.

        """

        # Compute RO gradients
        self.__SetGradientMat()
        # Compute timing related values

        # Prephaser
        ## Timing
        t_Prephaser = np.linspace(
            0,
            self.prephase_duration_x,
            num=int(self.matrix_heigh * self.prephase_fraction_x),
            endpoint=False,
        )
        ## Gradient
        G_Prephaser = np.ones_like(t_Prephaser) * self.prephase_amplitude

        # Additional unsampled gradient
        if self.unsampled_gradient and self.k_interleave:

            # Timing of the readout gradients
            t_tmp = np.empty([self.epi_factor + 1, self.matrix_heigh])
            t = self.prephase_duration_x
            for i in enumerate(range(self.epi_factor + 1)):
                t_tmp[i, :] = t + np.linspace(
                    0,
                    self.readout_duration,
                    num=self.matrix_heigh,
                    endpoint=False,
                )
                t += self.echospacing
            t_ReadOut = t_tmp.flatten()
            G_ReadOut = np.append(
                np.ones(self.matrix_heigh) * self.gradient_matrix[0],
                self.gradient_matrix * -1,
            )
        else:
            # Timing of the readout gradients
            t_tmp = np.empty([self.epi_factor, self.matrix_heigh])
            t = self.prephase_duration_x
            for i in enumerate(range(self.epi_factor)):
                t_tmp[i, :] = t + np.linspace(
                    0,
                    self.readout_duration,
                    num=self.matrix_heigh,
                    endpoint=False,
                )
                t += self.echospacing
            t_ReadOut = t_tmp.flatten()
            G_ReadOut = self.gradient_matrix

        # Set start time of first blip
        self.Time2FirstBlip = t_Prephaser[-1] + t_tmp[0, -1]
        # Combine all timing vectors
        t_combined = np.append(t_Prephaser, t_ReadOut)
        G_combined = np.append(G_Prephaser, G_ReadOut)

        return t_combined, G_combined

    def __SetGradientMat(self) -> None:
        """
        Initializes the gradient matrix based on the epi factor and starting polarity.

        The gradient amplitude is applied across the entire matrix, with alternating rows set to negative
        values depending on whether a "starting polarity" is specified.
        """
        self.gradient_matrix = self.gradient_amplitude * np.ones(
            (self.epi_factor, self.matrix_heigh)
        )
        if self.starting_polartiy:
            self.gradient_matrix[
                np.linspace(1, self.epi_factor, self.epi_factor) % 2 == 0, :
            ] = -self.gradient_amplitude
        else:
            self.gradient_matrix[
                np.linspace(1, self.epi_factor, self.epi_factor) % 2 != 0, :
            ] = -self.gradient_amplitude

    def SetSetCounter(self, matrix_width: int = None) -> None:
        """
        Initializes or updates the set counter array.

        The set counter is used to keep track of the readout polarities.

        """
        if matrix_width is None:
            self.setCounter = np.array([1 for line in range(self.matrix_width)])
            for line in range(self.epi_factor):
                if line % 2 == 0:
                    self.setCounter[line * self.shots : (line + 1) * self.shots] = 0
        else:
            epi_factor = matrix_width // self.shots
            self.setCounter = np.array([1 for line in range(matrix_width)])
            for line in range(epi_factor):
                if line % 2 == 0:
                    self.setCounter[line * self.shots : (line + 1) * self.shots] = 0

    def ProduceFlowImage(self, image: np.array = None) -> np.array:
        """
        Produces a flow image from the input data.

        """
        self.input_image = image
        # Initialize k-spaces with image or PSF
        if image is None:
            self.kSpace = np.ones(
                (self.matrix_width, self.matrix_heigh), dtype=complex
            )
        else:
            self.kSpace = ft_i2k(image)

        self.kSpace *= np.exp(self.gamma * 1.0j * 2 * np.pi * self.kSpace_phase)
        output_image = ft_k2i(self.kSpace)

        return output_image

    def ComputeMomentMat(self, orders: list = [0, 1, 2]) -> None:
        """
        Compute and return the moment matrix.

        This method calculates the moments for the specified orders and stores them in a matrix.

        """

        if self.flow_direction == "FE":
            moments = np.zeros([3, self.matrix_width, self.matrix_heigh])
            moments = self.__ComputeMoment_x(orders=orders)
        elif self.flow_direction == "PE":
            moments = np.zeros([1, self.matrix_width, self.matrix_heigh])
            moments = self.__ComputeMoment_y()

        else:
            raise ValueError(
                "Unexpected value for flow direction: '{}'".format(self.flow_direction)
            )
        return moments

    def __ComputeMoment_x(self, orders: list = [0, 1, 2]) -> np.array:
        """
        Compute moments of the k-space from readout gradients.

        This function calculates the zeroth-, first- and second-order moments of the
        k-space based on the given gradient matrix, flow velocity, acceleration,
        off-resonance, and timing map. It also accounts for prephasing effects if enabled.

        """
        moments = np.zeros([3, self.matrix_width, self.matrix_heigh])
        # Get timing and gradient map
        time_vector, gradient_vector = self.ProduceTimeAndGradientVector()
        for Shot in range(0, self.shots):
            for order in orders:
                # Offresonances
                if order == 0:
                    moment_tmp = integrate.cumulative_trapezoid(
                        gradient_vector * self.offresonance,
                        dx=self.ADC_dwelltime,
                        initial=0,
                    )
                # 1st order
                elif order == 1:
                    moment_tmp = (
                        integrate.cumulative_trapezoid(
                            gradient_vector * time_vector,
                            dx=self.ADC_dwelltime,
                            initial=0,
                        )
                        * self.flow_velocity
                    )
                # 2nd order
                elif order == 2:
                    moment_tmp = (
                        integrate.cumulative_trapezoid(
                            gradient_vector * time_vector**2 * 0.5,
                            dx=self.ADC_dwelltime,
                            initial=0,
                        )
                        * self.flow_acceleration
                    )
                # Sort current shot into matrix and remove unsampled gradient for k_repeat
                moments[order, Shot :: self.shots, :] = np.reshape(
                    moment_tmp[-self.matrix_heigh * self.epi_factor :],
                    (self.epi_factor, self.matrix_heigh),
                )
        # Flip data of negative gradient 
        if self.k_interleave:
            moments[:, self.setCounter == self.starting_polartiy, :] = moments[
                :, self.setCounter == self.starting_polartiy, ::-1
            ]
        else:
            moments[:, self.setCounter != self.starting_polartiy, :] = moments[
                :, self.setCounter != self.starting_polartiy, ::-1
            ]

        self.kSpace_phase = np.sum(np.array(moments), axis=0)

        return moments

    def __ComputeMoment_y(self) -> np.array:
        """
        Compute moments of the k-space from PE blips.

        This function calculates the first-order moments of the
        k-space based on the given gradient matrix, flow velocity and timing map.

        """        
        # Parameters of 1st blip
        t_s = self.prephase_duration_y + self.echospacing # Start time of 1st blip

        t_e = t_s + self.blip_duration  # End time

        self.delta_ETS = self.echospacing / self.shots * self.echo_time_shifting
        n_blips_before_center = self.epi_factor // 2
        m0_blip = self.blip_amplitude * self.blip_duration
        m0_prephas = (
            np.linspace(m0_blip / 2, -m0_blip / 2, self.shots, endpoint=False)
            + m0_blip * n_blips_before_center
        )
        m1_prephas = m0_prephas * self.prephase_duration_y

        moment_tmp = np.zeros((self.epi_factor, self.shots))
        for echo in range(self.epi_factor):
            for shot in range(self.shots):
                moment_tmp[echo, shot] = (
                    m1_prephas[shot]
                    - 0.5
                    * self.blip_amplitude
                    * self.M1_kth_Echo(t_s, t_e, echo + 1, shot + 1)
                ) * self.flow_velocity

        # Fill (N,N) matrix by repeating same PE-moment for every FE step
        moment = np.tile(moment_tmp.flatten(), (self.matrix_width, 1)).transpose()
        return moment

    def M1_kth_Echo(self, t_s, t_e, echo, shot):
        """
        Compute 1st order moments of current echo. 

        """
        moment = 0
        if echo >= 2:
            for i in np.linspace(0, echo - 2, echo - 2 + 1):
                moment += (
                    t_e**2
                    - t_s**2
                    + 2
                    * (i * self.echospacing + (shot - 1) * self.delta_ETS)
                    * (t_e - t_s)
                )
        return moment
    
    def GradientMomentSmoothing(self):
        """
        Main function to compute the effect of gradient moment smoothing (GMS).

        """
        kx = np.linspace(-1, 1, self.matrix_width, endpoint=False)
        # Step 1
        # Find the first moments of the N/2 th shot by calculating the numerical equation
        step1_x = kx[self.shots // 2 :: self.shots] * self.matrix_width / 2
        step1_y = self.moments_y[self.shots // 2 :: self.shots, self.matrix_width // 2]

        # Step 2
        # Find the first moments for the first echo of all shots
        Start = self.epi_factor // 2 * self.shots
        End = (self.epi_factor // 2 + 1) * self.shots
        step2_y = self.moments_y[Start:End, self.matrix_width // 2]

        # Step 3
        # Interpolate Step 1 quadratically and use the curve to determine the modified first moments for all Shots so that it is smooth at the center echo
        step3_x = kx[Start:End] * self.matrix_width / 2

        f3 = interpolate.interp1d(step1_x, step1_y, kind="quadratic")
        step3_y = f3(step3_x)

        # Step 4
        # Interpolate offset at ky=0 and subtract from step 3
        ky0 = kx[int(Start + (End - Start) / 2)] * self.matrix_width / 2
        step4_y = step3_y - f3(ky0)

        # Step 5
        # Subtract the actual first moment (step 2)
        # step5_y = step4_y - step2_y
        step5_y = step3_y - step2_y

        # Create moment matrix
        moment_GradSmooth = np.tile(
            np.transpose(np.tile(step5_y, (self.matrix_width, 1))), (self.epi_factor, 1)
        )

        return self.moments_y + moment_GradSmooth

    def Plot(
        self,
        phase: np.array = None,
        flyback: bool = False,
        compensated: bool = True,
        shuffeld: bool = None,
        ax: plt.axes = None,
    ) -> plt.Figure:
        """
        Plots the phase information of the object in 3D.

        """
        kx, ky = np.meshgrid(
            np.linspace(-1, 1, self.matrix_heigh, endpoint=False),
            np.linspace(-1, 1, self.matrix_width, endpoint=False),
        )
        nShot = self.shots
        if len(phase.shape) > 2:
            phase = np.sum(phase, axis=0)

        # convert unis
        phase = phase * self.trans_factor

        if not flyback:
            fig_para = self.fig_para.copy()
            fig_para["color"] = self.fig_para_c1
            fig_para_interleave = self.fig_para.copy()
            fig_para_interleave["color"] = self.fig_para_c2
        else:
            if shuffeld is None:
                if compensated:
                    fig_para = self.fig_para.copy()
                    fig_para["color"] = self.fig_para_c2
                    fig_para["linestyle"] = "--"
                    # fig_para["dashes"]=(0,(5, 10))
                    fig_para_interleave = self.fig_para.copy()
                    fig_para_interleave["color"] = self.fig_para_c2
                else:
                    fig_para = self.fig_para.copy()
                    fig_para["color"] = self.fig_para_c1
                    fig_para_interleave = self.fig_para.copy()
                    fig_para_interleave["color"] = self.fig_para_c1
                    fig_para_interleave["linestyle"] = "dashed"
            else:
                if compensated:
                    fig_para = self.fig_para.copy()
                    fig_para["color"] = self.fig_para_c1
                    fig_para_interleave = self.fig_para.copy()
                    fig_para_interleave["color"] = self.fig_para_c2
                else:
                    fig_para = self.fig_para.copy()
                    fig_para["color"] = self.fig_para_c2
                    fig_para["linestyle"] = "dashed"
                    fig_para_interleave = self.fig_para.copy()
                    fig_para_interleave["color"] = self.fig_para_c1
                    fig_para_interleave["linestyle"] = "dashed"

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d", **self.ax_para)
        else:
            fig = None

        for line in range(self.epi_factor):
            if IsEven(line):
                ax.plot_wireframe(
                    kx[line * nShot : (line + 1) * nShot : self.fig_para_skip],
                    ky[line * nShot : (line + 1) * nShot : self.fig_para_skip],
                    phase[line * nShot : (line + 1) * nShot : self.fig_para_skip],
                    **fig_para,
                )
            else:
                ax.plot_wireframe(
                    kx[line * nShot : (line + 1) * nShot : self.fig_para_skip],
                    ky[line * nShot : (line + 1) * nShot : self.fig_para_skip],
                    phase[line * nShot : (line + 1) * nShot : self.fig_para_skip],
                    **fig_para_interleave,
                )
        ax.plot(
            ky[:, self.matrix_heigh // 2],
            phase[:, self.matrix_heigh // 2],
            zs=kx[:, self.matrix_heigh // 2],
            zdir="x",
            color="k",
            linewidth=1,
        )
        ax.plot(ky[:, -1], phase[:, 0], zs=kx[:, 0], zdir="x", color="k", linewidth=1)
        ax.plot(
            ky[:, 0],
            phase[:, self.matrix_heigh - 1],
            zs=kx[:, -1],
            zdir="x",
            color="k",
            linewidth=1,
        )
        ax.view_init(azim=300, elev=20)
        return fig

    def AlternativePlot(
        self, ax=None, fig=None, moment=None, color="k", interleave=False
    ):
        """
        Plots the phase information of k-space as projection.

        """
        if moment is None:
            raise TypeError("Please assign the moment to plot.")
        moment = moment * self.trans_factor
        fig_para = self.ax_para.copy()

        fig_para["xticks"] = (
            0,
            self.matrix_heigh // 2,
            self.matrix_heigh,
        )
        fig_para["ylabel"] = fig_para["zlabel"]
        fig_para.pop("yticklabels")
        fig_para.pop("yticks")
        fig_para.pop("zlabel")
        fig_para.pop("zticks")
        plot_para = {
            "linewidth": 0.9,
        }
        if interleave:
            plot_para["linestyle"] = "dashed"
        else:
            plot_para["linestyle"] = "solid"

        line_para = {
            "linewidth": 0.9,
            "linestyle": "dotted",
            "color": "k",
            "alpha": 0.3,
        }
        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, **fig_para)
        set_label = [True,True]
        ax.axhline(0, **line_para)
        ax.axvline(self.matrix_heigh // 2, **line_para)
        for shot in range(self.shots):
            for line in range(self.epi_factor):
                if (IsEven(line) and not interleave) or (
                    not IsEven(line) and interleave
                ):
                    cur_label = ("$k_{repeated,c}$" if interleave else "$k_{initial,uc}$") if set_label[0] else "_nolegend_"
                    ax.plot(
                        moment[1, shot :: self.shots, :][line, :],
                        color=self.fig_para_c1,
                        label=cur_label,
                        **plot_para,
                    )
                    set_label[0] = False
                else:
                    cur_label = ("$k_{repeated,uc}$" if interleave else "$k_{initial,c}$") if set_label[1] else "_nolegend_"
                    ax.plot(
                        moment[1, shot :: self.shots, :][line, :],
                        color=self.fig_para_c2,
                        label=cur_label,
                        **plot_para,
                    )
                    set_label[1] = False

        ax.legend()
        return fig,ax


class EPISimulator(__BaseSimulator):
    """
    Simulation class for regular EPI.

    """

    def __init__(self, preset_name: str = None):
        """
        Initializes the EPI simulator with a given preset name.

        """
        super().__init__()
        self.set_preset(preset_name)

    def Compute(self, image: np.array = None) -> None:
        """
        Computes the moments and produces an output image.

        """
        self.apply_preset()
        self.check_variables()
        self.moments_x = self.ComputeMomentMat(orders=[1, 2])
        self.output_image = self.ProduceFlowImage(image=image)

    def PlotPhase(self, show: bool = False, moment: np.array = None) -> None:
        """
        Plots the phase of the sequence.

        """
        if moment is None:
            self.PhaseFigure_x = self.Plot(phase=self.moments_x)
        else:
            self.PhaseFigure_x = self.Plot(phase=moment)
        ax = self.PhaseFigure_x.get_axes()
        for axis in ax:
            axis.set_title('Regular EPI')
            axis.set_box_aspect(None, zoom=0.85)

        if show:
            plt.show()
        else:
            plt.close()


class FlybackSimulator(__BaseSimulator):
    """
    Simulation class for modified interleaved flyback miFB.

    """

    def __init__(self, preset_name: str = None, unsampled_gradient: bool = False):
        # Class specific variables
        self.timing_map_interleave = None
        self.moments_interleave = None
        self.moments_comp = None
        self.moments_uncomp = None
        self.image_comp = None
        self.image_uncomp = None

        super().__init__()
        self.set_preset(preset_name)
        self.unsampled_gradient = unsampled_gradient

    def IncreaseMatrixWidth(self) -> None:
        """
        Increases the width of the matrix for k_repeated and unsampled gradient.

        """
        self.matrix_width += self.shots
        self.resolve_dependencies()

    def DecreaseMatrixWidth(self) -> None:
        """
        Decreases the width of the matrix for k_initial.

        :return: None
        """
        self.matrix_width -= self.shots
        self.resolve_dependencies()

    def ReorderKspaces(self) -> None:
        """
        Reorders moments based on set counter and interleave data.

        """
        self.SetSetCounter(matrix_width=self.matrix_width)

        self.moments_uncomp = np.copy(self.moments_x)
        self.moments_uncomp[:, self.setCounter == 1, :] = self.moments_interleave[
            :, self.setCounter == 1, :
        ]

        self.moments_comp = np.copy(self.moments_x)
        self.moments_comp[:, self.setCounter == 0, :] = self.moments_interleave[
            :, self.setCounter == 0, :
        ]

    def ProduceFlybackImage(self, image=None):
        """
        Produces a flow image for both reordered k-spaces.

        """
        self.SetCurrentKSpace(np.sum(np.array(self.moments_comp), axis=0))
        image_comp = self.ProduceFlowImage(image=image)

        self.SetCurrentKSpace(np.sum(np.array(self.moments_uncomp), axis=0))
        image_uncomp = self.ProduceFlowImage(image=image)

        return image_comp, image_uncomp

    def SetCurrentKSpace(self, kSpace: np.array = None) -> None:
        """
        Sets the current k-space.

        """
        self.kSpace_phase = kSpace

    def Compute_x(self, image: np.array = None) -> None:
        """
        Main function to compute the moments of the readout gradients.

        This function is responsible for computing the moment matrices, 
        and reordering of k-spaces.

        """
        self.apply_preset()
        self.check_variables()
        self.SetSetCounter()

        # k_initial
        self.moments_x = self.ComputeMomentMat(orders=[1])
        self.k_interleave = True
        # k_reverse
        self.moments_interleave = self.ComputeMomentMat(orders=[1])
        # Reorder k-spaces
        self.ReorderKspaces()
        self.image_comp, self.image_uncomp = self.ProduceFlybackImage(image=image)

    def Compute_y(self) -> None:
        """
        Main function to compute and display the moments of PE blips.

        This function is responsible for computing the moment matrices from the PE blips and calling the GMs method.

        """
        self.apply_preset()
        self.set_variable("flow_direction", "PE")
        self.check_variables()
        self.ProduceTimeAndGradientVector()
        self.moments_y = self.ComputeMomentMat()
        self.moment_GradSmooth = self.GradientMomentSmoothing()

    def Plot_xPhase(self, show: bool = False, flyback_image: str = None) -> None:
        """
        Plots the phase of the x-moments.

        """
        if self.figdir is None:
            print("Warning: No output direction specified. Skipping saving images.")
            self.save_figures = None

        if flyback_image is not None:
            if flyback_image == "compensated":
                self.PhaseFigure_x = self.Plot(
                    phase=self.moments_comp,
                    flyback=True,
                    compensated=True,
                )
            else:
                self.PhaseFigure_x = self.Plot(
                    phase=self.moments_uncomp,
                    flyback=True,
                    compensated=False,
                )
        else:
            fig = plt.figure(figsize=(8, 6))
            fig, ax = plt.subplots(2,2,subplot_kw={'projection': '3d'},figsize=(8, 6))
            ax = ax.flatten()
            self.PhaseFigure_x = self.Plot(
                phase=self.moments_x,
                flyback=True,
                compensated=True,
                shuffeld=True,
                ax=ax[0]
            )
            ax[0].set_title('$k_{initial}$')
            self.PhaseFigure_x = self.Plot(
                phase=self.moments_interleave,
                flyback=True,
                compensated=False,
                shuffeld=True,
                ax=ax[1]
            )
            ax[1].set_title('$k_{repeated}$')
            self.PhaseFigure_x = self.Plot(
                phase=self.moments_comp,
                flyback=True,
                compensated=True,
                ax=ax[2]
            )
            ax[2].set_title('$k_{comp}$')
            self.PhaseFigure_x = self.Plot(
                phase=self.moments_uncomp,
                flyback=True,
                compensated=False,
                ax=ax[3]
            )
            ax[3].set_title('$k_{uncomp}$')
            for axes in ax:
                axes.set( **self.ax_para)
        if show:
            plt.show()
        else:
            plt.close()

    def Plot_xProjection(self, show: bool = False) -> None:
        """
        Plots the phase of the x-moments as projection.

        """
        if self.figdir is None:
            print("Warning: No output direction specified skipping saving images.")
            self.save_figures = False

        fig, ax = self.AlternativePlot(moment=self.moments_x, interleave=False)
        fig, ax = self.AlternativePlot(
            moment=self.moments_interleave, interleave=True, ax=ax, fig=fig
        )
        if self.save_figures:
            fig.savefig(
                os.path.join(
                    self.figdir,
                    "simulation_moment_Interleave1_projection" + self.saveas,
                ),
                dpi=300,
                bbox_inches="tight",
            )
        if show:
            plt.show()
        else:
            plt.close()

    def Plot_yPhase(self, show: bool = False) -> None:
        """
        Plots the phase of the y-moments.

        """
        fig_para = {
            "nETS,nGMS": {
                "label": "nETS,nGMs",
                "marker": "",
                "linestyle": "dashed",
                "color": "k",
                "alpha": 0.5,
                "linewidth": 0.9,
            },
            "ETS,nGMS": {
                "label": "ETS,nGMs",
                "marker": "",
                "linestyle": "dotted",
                "color": "k",
                "alpha": 0.5,
                "linewidth": 0.9,
            },
            "ETS,GMS": {
                "label": "ETS,GMs",
                "marker": "",
                "linestyle": "solid",
                "color": self.fig_para_c3,
                "linewidth": 0.9,
            },
        }
        if self.figdir is None:
            print("Warning: No output direction specified skipping saving images.")
            self.save_figures = False
        kx = np.linspace(
            -self.matrix_width // 2,
            self.matrix_width // 2,
            self.matrix_width,
            endpoint=False,
        )
        self.PhaseFigure_y, ax = plt.subplots(1, 1,figsize=(5,5))
        if self.echo_time_shifting == 1:
            self.set_variable("echo_time_shifting", 0)
            moment_nETS = self.ComputeMomentMat()
            ax.plot(
                kx,
                moment_nETS[:, self.matrix_width // 2] * self.trans_factor,
                **fig_para["nETS,nGMS"],
            )
            ax.plot(
                kx,
                self.moments_y[:, self.matrix_width // 2] * self.trans_factor,
                **fig_para["ETS,nGMS"],
            )
            ax.plot(
                kx,
                self.moment_GradSmooth[:, self.matrix_width // 2] * self.trans_factor,
                **fig_para["ETS,GMS"],
            )
        else:
            self.set_variable("echo_time_shifting", 1)
            moment_ETS = self.ComputeMomentMat()
            ax.plot(
                kx,
                moment_ETS[:, self.matrix_width // 2] * self.trans_factor,
                **fig_para["ETS,nGMS"],
            )
            ax.plot(
                kx,
                self.moments_y[:, self.matrix_width // 2] * self.trans_factor,
                **fig_para["nETS,nGMS"],
            )
            ax.plot(
                kx,
                self.moment_GradSmooth[:, self.matrix_width // 2] * self.trans_factor,
                **fig_para["ETS,GMS"],
            )
        ax.legend()
        ax.set_xlabel(self.ax_para["ylabel"])
        ax.set_ylabel(self.ax_para["zlabel"])
        ax.set_xticks(np.array(self.ax_para["yticks"])*(self.matrix_width // 2))
        ax.set_xticklabels(self.ax_para["yticklabels"])
        ax.set_yticks([-1.75, -1.25, -0.75, -0.25, 0.25])
        if self.save_figures:
            self.PhaseFigure_y.savefig(
                os.path.join(self.figdir, "simulation_moment_GMs.svg")
            )
        if show:
            plt.show()
        else:
            plt.close()