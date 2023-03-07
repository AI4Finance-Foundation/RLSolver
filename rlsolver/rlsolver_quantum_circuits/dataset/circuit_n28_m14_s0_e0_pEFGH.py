import cirq
import numpy as np

QUBIT_ORDER = [
    cirq.GridQubit(1, 5),
    cirq.GridQubit(1, 6),
    cirq.GridQubit(2, 4),
    cirq.GridQubit(2, 5),
    cirq.GridQubit(2, 6),
    cirq.GridQubit(2, 7),
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(3, 7),
    cirq.GridQubit(4, 2),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(4, 7),
    cirq.GridQubit(5, 2),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
    cirq.GridQubit(6, 2),
    cirq.GridQubit(6, 3),
    cirq.GridQubit(6, 4),
    cirq.GridQubit(6, 5),
    cirq.GridQubit(7, 3),
    cirq.GridQubit(7, 4),
]

CIRCUIT = cirq.Circuit(
    [
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
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
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=3.8865873271372564).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-4.029178772626372).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.05145215307733).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.6675374774973344).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=0.8693959871027742).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.5809728937821895).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.07013891860451749).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.09433754708706699).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.660146660745897).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.565089226594822).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-3.1455232066056915).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=3.5262953599473446).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.5124113980328648).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.5548628633703563).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.7081899981850245).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.8842513870221531).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.0402901334038208).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.2704822022121596).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.2821664097933372).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.1519683636480256).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.1018534267102214).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.1798141095619943).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.59182423935832, phi=0.5095208431994713).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.406681628061114).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.264090182571999).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=4.141737645019865).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-5.525652320599861).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.1118243782923773).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.4002474716129623).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.8218346808699086).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.8460333093524581).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=3.406545499174985).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-3.50160293332606).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.510370594218442).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.1295984408767894).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-5.107675813207988).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=5.150127278545479).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.497030521004973).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.3209691321678445).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.15501230573908462).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.07517976306925431).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=1.4798871549098012).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.610085201055113).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=6.800246368359428).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-6.722285685507655).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-17.07728575198494).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=17.085543130774045).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-16.30639128029667).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=16.245140620956132).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=7.268041060268155).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-7.032434485579018).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=4.480876907223065).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-4.7973193555210685).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=18.201935610761314).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-18.19367823197221).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=16.408975290236828).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-16.470225949577365).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=-7.379726055912839).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=7.6153326306019755).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-5.8609025465851285).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=5.544460098287125).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-19.522557672224195).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=19.61709690587081).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-19.466235154980904).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=19.452218392199153).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-16.86365710820967).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=16.515844503409756).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-5.0636417854867775).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=4.921365331976502).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-16.99266260171233).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=16.95272957943951).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-10.300317334985465).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=10.489798112305557).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-16.935559040199575).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=17.018814476659895).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-17.65620553132214).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=17.688640528946085).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=13.72138435193195).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-12.285441868370123).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=17.909821610830367).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-17.85485918787798).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=14.408398823695533).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-14.970868747719962).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.505206014385737, phi=0.5177720559789512).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5588791081427968, phi=0.559649620487243).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5907035825834708, phi=0.5678223287662552).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5296321276792553, phi=0.537761951313038).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.5306030283605572, phi=0.5257102080843467).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
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
            cirq.FSimGate(theta=1.53597466118183, phi=0.5584919013659856).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.385350861888917, phi=0.5757363921651084).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=21.675242779969807).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-21.580703546323193).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=20.441134318784886).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-20.455151081566637).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=19.322384645379532).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-19.670197250179445).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=7.567246854917207).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-7.709523308427483).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=14.883941980954871).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-14.92387500322769).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=9.244163795666879).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-9.054683018346786).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=14.03202226075948).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-13.94876682429916).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=14.094665893757096).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-14.062230896133155).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-15.384431193528544).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=16.82037367709037).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-18.097545706581567).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=18.152508129533953).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-12.521705657743373).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=11.959235733718943).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
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
                cirq.GridQubit(3, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
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
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-7.811835872200318).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=7.81967986206747).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-7.615465998041652).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=7.600763065330671).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=39.909529911684075).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-40.171428733227174).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=38.277959915149545).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-38.040241149679936).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=11.883270567757798).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-12.169262658895612).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=14.993690269805821).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-14.989527131529702).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=26.69077423311042).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-26.640419634864774).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=30.166300933501873).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-30.177005939701825).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=19.88784916538296).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-19.533502953512624).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=30.950507066341867).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-31.18491014533392).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=20.20391102349164).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-20.06411500554498).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5684106752459124, phi=0.5414007317481024).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.6152322695478165, phi=0.5160697976136035).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5909523830878005, phi=0.5244700889486827).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5245711693927642, phi=0.4838906581970925).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.5542388360689805, phi=0.5186534637665338).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5109427139358562, phi=0.4939388316289224).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.57896484905089, phi=0.5081656554152614).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=8.035799530009061).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-8.027955540141907).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=6.217196884955984).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-6.231899817666965).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-39.198437626672444).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=38.936538805129345).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-35.53420813044422).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=35.77192689591383).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-17.246915283138684).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=16.96092319200087).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-18.421895502761547).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=18.426058641037667).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-27.805272272134683).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=27.85562687038033).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-31.524503153421914).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=31.513798147221962).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-17.072738290280107).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=17.427084502150443).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-28.839463497649664).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=28.60506041865761).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-20.63084996378606).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=20.770645981732716).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=16.905347283613576).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-17.047938729102693).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-21.5795920877914).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=20.195677412211406).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=16.21293450723508).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-15.924511413914495).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.25463964505188).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.2304410165693302).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-14.748995191759326).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=14.653937757608254).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-14.76941602488768).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=15.150188178229332).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=11.111481420249536).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-11.069029954912043).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-3.497924274572867).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=3.6739856634099954).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-11.269315813491886).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=11.499507882300225).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-16.625704929925643).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=16.495506883780333).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-19.94700078316722).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=20.024961466018993).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.59182423935832, phi=0.5095208431994713).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-15.425441584537436).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=15.282850139048316).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=23.66987757973394).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-25.053792255313933).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-17.455362898424685).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=17.743785991745266).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=4.146613244526306).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-4.170811873008856).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=15.495394030188416).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-15.590451464339493).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=14.134263412500427).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-13.753491259158777).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-16.731568631490386).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=16.77402009682788).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=5.286764797392815).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-5.110703408555686).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=10.384037985827149).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-10.15384591701881).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=16.823425675042106).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-16.953623721187416).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=25.64539372481643).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-25.567433041964655).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-56.133565621413084).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=56.14182300020218).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-55.362671149725216).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=55.30142049038468).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=24.718321278531224).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-24.48271470384209).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=13.779991161849068).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-14.09643361014707).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=57.25821548018944).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-57.24995810140036).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=55.46525515966539).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-55.52650581900591).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=-24.83000627417591).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=25.065612848865047).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-15.160016801211128).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=14.843574352913127).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-52.069457563414176).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=52.16399679706079).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-49.6883564825149).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=49.67433971973315).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-41.97126559569901).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=41.6234529908991).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-12.967888901918446).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=12.825612448408174).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-41.63531537647063).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=41.59538235419781).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-25.643855855118186).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=25.833336632438275).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-40.648300389495404).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=40.73155582595572).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-39.97407974242405).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=40.00651474004799).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=34.64439142483994).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-33.20844894127811).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=45.80716437470755).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-45.75220195175516).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=34.15457562393565).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-34.717045547960076).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.505206014385737, phi=0.5177720559789512).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5588791081427968, phi=0.559649620487243).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5907035825834708, phi=0.5678223287662552).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5296321276792553, phi=0.537761951313038).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.5306030283605572, phi=0.5257102080843467).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
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
            cirq.FSimGate(theta=1.53597466118183, phi=0.5584919013659856).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.385350861888917, phi=0.5757363921651084).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=54.22214267115979).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-54.127603437513166).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=50.66325564631889).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-50.677272409100624).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=44.42999313286887).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-44.777805737668785).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=15.47149397134888).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-15.613770424859151).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=39.52659475571318).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-39.566527777985996).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=24.587702315799596).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-24.398221538479504).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=37.744763610055315).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-37.661508173594996).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=36.412540104859005).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-36.38010510723507).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-36.307438266436534).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=37.743380749998366).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-45.994888470458754).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=46.04985089341114).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-32.26788245798348).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=31.70541253395906).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
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
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-14.786171563169507).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=14.79401555303666).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-14.589801689011258).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=14.57509875630028).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=81.75554405750046).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-82.01744287904356).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=79.65901834823408).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-79.42129958276446).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=26.296897662428034).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-26.58288975356585).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=31.732095928132054).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-31.727932789855927).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=57.377851273375434).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-57.327496675129794).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=64.10806796288578).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-64.11877296908573).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=41.27581195102239).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-40.921465739152055).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=62.81441437927518).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-63.04881745826723).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=43.45169666005602).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-43.31190064210937).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5684106752459124, phi=0.5414007317481024).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.6152322695478165, phi=0.5160697976136035).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5909523830878005, phi=0.5244700889486827).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5245711693927642, phi=0.4838906581970925).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.5542388360689805, phi=0.5186534637665338).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5109427139358562, phi=0.4939388316289224).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.57896484905089, phi=0.5081656554152614).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=15.010135220978253).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-15.002291231111096).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=13.19153257592559).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-13.206235508636567).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-81.04445177248883).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=80.78255295094573).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-76.91526656352876).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=77.15298532899838).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-31.660542377808927).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=31.37455028667111).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-35.16030116108778).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=35.16446429936389).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-58.49234931239971).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=58.54270391064535).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-65.46627018280581).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=65.45556517660587).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-38.460701075919545).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=38.81504728778988).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-60.703370810582975).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=60.46896773159093).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-43.87863560035044).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=44.01843161829709).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
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
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=29.924107240089892).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-30.06669868557901).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-41.10773202250547).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=39.72381734692547).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=31.556473027367385).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-31.268049934046807).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-4.579418208708278).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=4.555219580225728).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-26.83784372277276).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=26.742786288621687).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-26.393308843169667).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=26.774080996511316).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=22.735374238531932).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-22.69292277319444).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-6.2876585509607095).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=6.463719939797837).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-21.49834149357995).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=21.728533562388286).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-31.96924345005795).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=31.83904540391264).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-38.79214813962422).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=38.870108822475984).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.59182423935832, phi=0.5095208431994713).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-28.444201541013754).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=28.301610095524637).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=43.198017514448).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-44.581932190028).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-32.798901418556994).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=33.08732451187758).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=6.4713918081827035).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-6.495590436665254).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=27.584242561201844).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-27.679299995352917).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=25.75815623078241).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-25.37738407744076).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-28.355461449772783).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=28.397912915110275).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=8.076499073780658).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-7.900437684943529).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=20.613063665915213).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-20.382871597106877).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=32.166964195174415).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-32.297162241319725).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=44.49054108127342).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-44.41258039842165).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-95.18984549084122).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=95.19810286963032).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-94.41895101915375).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=94.35770035981321).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=42.1686014967943).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-41.93299492210517).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=23.079105416475066).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-23.39554786477307).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=96.31449534961759).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-96.30623797082849).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=94.52153502909391).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-94.58278568843446).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=-42.280286492439).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=42.515893067128125).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-24.459131055837133).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=24.142688607539128).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-84.61635745460414).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=84.71089668825074).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-79.91047781004889).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=79.89646104726714).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-67.07887408318834).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=66.73106147838843).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-20.872136018350115).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=20.72985956483984).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-66.27796815122893).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=66.23803512895613).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-40.9873943752509).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=41.176875152571).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-64.36104173879124).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=64.44429717525156).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-62.29195395352596).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=62.3243889511499).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=55.56739849774794).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-54.13145601418609).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=73.70450713858472).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-73.64954471563235).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=53.900752424175764).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-54.4632223482002).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.505206014385737, phi=0.5177720559789512).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5588791081427968, phi=0.559649620487243).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5907035825834708, phi=0.5678223287662552).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5296321276792553, phi=0.537761951313038).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.5306030283605572, phi=0.5257102080843467).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
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
            cirq.FSimGate(theta=1.53597466118183, phi=0.5584919013659856).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.385350861888917, phi=0.5757363921651084).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=86.76904256234975).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-86.67450332870315).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=80.88537697385289).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-80.89939373663464).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=69.5376016203582).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-69.88541422515812).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=23.375741087780543).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-23.51801754129082).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=64.16924753047148).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-64.20918055274429).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=39.931240835932314).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-39.741760058612215).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=61.45750495935115).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-61.37424952289083).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=58.730414315960914).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-58.69797931833698).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-57.23044533934453).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=58.666387822906366).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-73.89223123433592).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=73.9471936572883).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-52.0140592582236).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=51.45158933419917).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-21.7605072541387).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=21.768351244005856).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-21.564137379980863).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=21.54943444726988).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=123.60155820331686).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-123.86345702485994).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=121.04007678131862).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-120.80235801584898).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=40.71052475709828).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-40.99651684823609).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=48.47050158645827).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-48.46633844818216).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=88.06492831364045).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-88.01457371539482).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=98.04983499226967).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-98.06053999846961).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=62.66377473666182).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-62.309428524791485).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=94.6783216922085).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-94.91272477120054).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=66.6994822966204).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-66.55968627867375).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5684106752459124, phi=0.5414007317481024).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.6152322695478165, phi=0.5160697976136035).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5909523830878005, phi=0.5244700889486827).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5245711693927642, phi=0.4838906581970925).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.5542388360689805, phi=0.5186534637665338).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5109427139358562, phi=0.4939388316289224).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.57896484905089, phi=0.5081656554152614).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=21.984470911947447).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-21.976626922080293).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=20.165868266895195).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-20.180571199606177).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-122.89046591830522).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=122.62856709676213).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-118.29632499661328).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=118.5340437620829).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-46.07416947247916).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=45.78817738134135).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-51.89870681941401).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=51.902869957690115).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-89.17942635266473).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=89.22978095091037).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-99.4080372121897).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=99.39733220598976).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-59.848663861558975).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=60.20301007342931).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-92.56727812351627).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=92.33287504452423).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-67.12642123691484).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=67.26621725486149).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=42.94286719656621).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-43.08545864205533).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-60.63587195721954).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=59.25195728163954).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=46.900011547499695).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-46.611588454179106).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-6.904196772364674).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=6.8799981438821245).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-38.92669225378619).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=38.83163481963512).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-38.017201661451644).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=38.397973814793296).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=34.359267056814325).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-34.31681559147683).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-9.07739282734855).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=9.25345421618568).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-31.727367173668014).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=31.957559242476353).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-47.31278197019025).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=47.18258392404494).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-57.63729549608121).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=57.71525617893297).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5930079664614663, phi=0.5355369376884288).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.59182423935832, phi=0.5095208431994713).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5862983338115253, phi=0.5200148508319427).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5286450573669954, phi=0.5113953905811602).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.565622495548066, phi=0.5127256481964074).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5289739216684795, phi=0.5055240639761313).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5384796865621224, phi=0.5293381306162406).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.4727562833004122, phi=0.4552443293379814).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5346175385256955, phi=0.5131039467233695).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5169062231051558, phi=0.46319906116805815).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5705414623224259, phi=0.4791699064049766).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-41.46296149749007).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=41.32037005200095).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=62.726157449162066).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-64.11007212474206).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-48.1424399386893).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=48.43086303200989).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=8.796170371839102).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-8.82036900032165).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=39.67309109221528).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-39.76814852636635).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=37.3820490490644).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-37.00127689572275).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-39.97935426805518).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=40.021805733392675).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=10.8662333501685).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-10.69017196133137).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=30.842089346003284).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-30.611897277194945).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=47.51050271530672).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-47.64070076145203).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=63.33568843773041).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-63.257727754878644).on(cirq.GridQubit(6, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-134.24612536026936).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=134.25438273905846).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-133.47523088858233).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=133.4139802292418).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=59.61888171505737).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-59.38327514036824).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=32.37821967110107).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-32.69466211939908).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5233234922971755, phi=0.6681144400379464).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5644541080112795, phi=0.5439498075085039).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
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
            cirq.FSimGate(theta=1.5376836849431186, phi=0.46265685930712236).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4749003996237158, phi=0.4353609222411594).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=135.37077521904573).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-135.36251784025663).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=133.57781489852246).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-133.639065557863).on(cirq.GridQubit(2, 6)),
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
            cirq.Rz(rads=-59.73056671070207).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=59.96617328539119).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-33.75824531046313).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=33.441802862165126).on(cirq.GridQubit(7, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
        ),
    ]
)

