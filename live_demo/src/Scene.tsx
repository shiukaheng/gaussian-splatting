import { OrbitControls, Splat } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useControls } from 'leva'
import { Perf } from 'r3f-perf'
import { useRef } from 'react'
import { BoxGeometry, Mesh, MeshBasicMaterial } from 'three'
import { Cube } from './components/Cube'
import { Plane } from './components/Plane'
import { Sphere } from './components/Sphere'

function Scene() {
  const { performance } = useControls('Monitoring', {
    performance: false,
  })

  const { animate } = useControls('Cube', {
    animate: true,
  })

  return (
    <>
      {performance && <Perf position='top-left' />}

      <OrbitControls makeDefault />

      <Splat src="/output/5127208a-4/point_cloud/iteration_7000/point_cloud.ply" scale={1} />
    </>
  )
}

export { Scene }
