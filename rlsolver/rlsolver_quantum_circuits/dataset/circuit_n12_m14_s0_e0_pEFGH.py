import cirq
import numpy as np

QUBIT_ORDER = [
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
]

CIRCUIT = cirq.Circuit(
    [
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.8693959871027742).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.5809728937821895).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-3.1455232066056915).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=3.5262953599473446).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.0402901334038208).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.2704822022121596).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.1118243782923773).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.4002474716129623).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.510370594218442).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.1295984408767894).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.15501230573908462).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.07517976306925431).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=7.95878242287787).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-7.774843741231972).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-14.076559546984539).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=14.218497333398785).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=6.7102990377713985).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-6.8557924692300185).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-11.869086784143517).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=11.992514358506712).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.4540157696942893).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.130087599403273).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=5.854589755013336).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-6.755719773321365).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-7.370403665363804).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=7.554342347009701).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=15.859387461711556).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-15.71744967529731).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-8.338843337248123).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=8.193349905789502).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=12.26637631755088).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-12.142948743187686).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-4.874679037269875).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=5.198607207560892).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-5.948000038116973).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=5.046870019808944).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-10.300317334985465).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=10.489798112305557).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-16.935559040199575).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=17.018814476659895).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-17.65620553132214).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=17.688640528946085).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=13.72138435193195).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-12.285441868370123).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5124128267683938, phi=0.5133142626030278).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=9.244163795666879).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-9.054683018346786).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=14.03202226075948).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-13.94876682429916).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=14.094665893757096).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-14.062230896133155).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-15.384431193528544).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=16.82037367709037).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=39.909529911684075).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-40.171428733227174).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=38.277959915149545).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-38.040241149679936).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=11.883270567757798).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-12.169262658895612).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=14.993690269805821).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-14.989527131529702).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.47511091993527, phi=0.538612093835262).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-39.198437626672444).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=38.936538805129345).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-35.53420813044422).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=35.77192689591383).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-17.246915283138684).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=16.96092319200087).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-18.421895502761547).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=18.426058641037667).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=16.21293450723508).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-15.924511413914495).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-14.76941602488768).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=15.150188178229332).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-11.269315813491886).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=11.499507882300225).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-17.455362898424685).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=17.743785991745266).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=14.134263412500427).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-13.753491259158777).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=10.384037985827149).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-10.15384591701881).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=23.76727665574203).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-23.583337974096132).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-48.01832657636844).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=48.160264362782684).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=22.053837557903705).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-22.199330989362323).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-43.48607524987143).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=43.609502824234625).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=9.42835146066348).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-9.104423290372464).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=17.47848257329532).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-18.37961259160335).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-23.178897898227966).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=23.362836579873864).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=49.801154491095446).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-49.6592167046812).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-23.682381857380427).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=23.53688842592181).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=43.88336478327879).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-43.7599372089156).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-11.849014728239068).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=12.172942898530083).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-17.571892856398957).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=16.670762838090926).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-25.643855855118186).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=25.833336632438275).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-40.648300389495404).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=40.73155582595572).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-39.97407974242405).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=40.00651474004799).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=34.64439142483994).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-33.20844894127811).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5124128267683938, phi=0.5133142626030278).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=24.587702315799596).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-24.398221538479504).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=37.744763610055315).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-37.661508173594996).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=36.412540104859005).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-36.38010510723507).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-36.307438266436534).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=37.743380749998366).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=81.75554405750046).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-82.01744287904356).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=79.65901834823408).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-79.42129958276446).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=26.296897662428034).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-26.58288975356585).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=31.732095928132054).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-31.727932789855927).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.47511091993527, phi=0.538612093835262).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-81.04445177248883).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=80.78255295094573).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-76.91526656352876).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=77.15298532899838).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-31.660542377808927).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=31.37455028667111).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-35.16030116108778).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=35.16446429936389).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=31.556473027367385).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-31.268049934046807).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-26.393308843169667).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=26.774080996511316).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-21.49834149357995).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=21.728533562388286).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-32.798901418556994).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=33.08732451187758).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=25.75815623078241).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-25.37738407744076).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=20.613063665915213).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-20.382871597106877).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=39.57577088860619).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-39.391832206960295).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-81.96009360575232).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=82.10203139216657).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=37.39737607803601).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-37.54286950949463).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-75.10306371559935).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=75.22649128996255).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=16.402687151632673).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-16.078758981341654).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=29.102375391577308).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-30.00350540988534).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-38.98739213109213).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=39.17133081273803).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=83.74292152047934).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-83.6009837340651).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-39.025920377512726).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=38.88042694605411).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=75.50035324900671).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-75.3769256746435).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-18.823350419208257).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=19.147278589499276).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-29.19578567468094).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=28.294655656372914).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-40.9873943752509).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=41.176875152571).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-64.36104173879124).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=64.44429717525156).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-62.29195395352596).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=62.3243889511499).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=55.56739849774794).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-54.13145601418609).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.589821065740506, phi=0.5045391214115686).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5472406430590444, phi=0.5216932173558055).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5124128267683938, phi=0.5133142626030278).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5707871303628709, phi=0.5176678491729374).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=39.931240835932314).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-39.741760058612215).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=61.45750495935115).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-61.37424952289083).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=58.730414315960914).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-58.69797931833698).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-57.23044533934453).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=58.666387822906366).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=123.60155820331686).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-123.86345702485994).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=121.04007678131862).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-120.80235801584898).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=40.71052475709828).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-40.99651684823609).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=48.47050158645827).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-48.46633844818216).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.4668587973263782, phi=0.4976074601121169).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.47511091993527, phi=0.538612093835262).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.603651215218248, phi=0.46649538437100246).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.6160334279232749, phi=0.4353897326147861).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-122.89046591830522).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=122.62856709676213).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-118.29632499661328).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=118.5340437620829).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-46.07416947247916).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=45.78817738134135).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-51.89870681941401).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=51.902869957690115).on(cirq.GridQubit(4, 6)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=46.900011547499695).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-46.611588454179106).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-38.017201661451644).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=38.397973814793296).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-31.727367173668014).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=31.957559242476353).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-48.1424399386893).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=48.43086303200989).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=37.3820490490644).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-37.00127689572275).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=30.842089346003284).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-30.611897277194945).on(cirq.GridQubit(5, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.Rz(rads=55.38426512147036).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-55.20032643982446).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-115.90186063513623).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=116.04379842155048).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=52.740914598168324).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-52.88640802962693).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-106.72005218132728).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=106.84347975569048).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=23.377022842601868).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-23.05309467231085).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=40.72626820985929).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-41.627398228167316).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.2947043217999283, phi=0.4859467238431821).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.541977006124425, phi=0.6073798124875975).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5138652502397498, phi=0.47710618607286504).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5849169442855044, phi=0.54346233613361).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5398075246432927, phi=0.5174515645943538).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.4593314109380113, phi=0.5230636172671492).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-54.795886363956285).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=54.97982504560218).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=117.68468854986324).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-117.54275076344899).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-54.36945889764503).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=54.22396546618641).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=107.11734171473461).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-106.9939141403714).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-25.797686110177455).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=26.12161428046847).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-40.81967849296293).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=39.918548474654905).on(cirq.GridQubit(5, 6)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
        ),
    ]
)

